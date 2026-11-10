#!/usr/bin/env bash
# Roda todos os experimentos de desempenho e grava os CSVs em results/raw/.
#
# Uso: scripts/run_experiments.sh [experimento ...]
#   Sem argumentos, roda todos. Experimentos: size omp_threads affinity
#   omp_temporal vectorization cuda_blocks cuda_coarsen cuda_size
#   QUICK=1 reduz tamanhos e repetições (para testar o script).
#
# Cada linha dos CSVs tem uma coluna "setting" antes das colunas dos
# executáveis (afinidade, build etc.). Os resultados de cada configuração são
# conferidos com o compare, e os erros vão para results/raw/validation_runs.csv.

set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH

BUILD=build
RAW=results/raw
mkdir -p "$RAW"

# Tamanhos e iterações: cerca de 1e9 atualizações por execução.
# Conjunto de trabalho (2 matrizes double): 1 MB, 4 MB, 16 MB, 64 MB, 256 MB, 1 GB.
SIZES=(256 512 1024 2048 4096 8192)
declare -A ITERS=([256]=15000 [512]=4000 [1024]=1000 [2048]=250 [4096]=60 [8192]=16)
REPS=5
THREADS=(1 2 4 6 8 10 12 14 16)
OMP_TEMPORAL=(--variant temporal --tile 128 --tsteps 16)

if [ "${QUICK:-0}" = 1 ]; then
	SIZES=(256 1024 4096)
	REPS=2
	THREADS=(1 4 16)
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

HEADER="setting,version,variant,prec,N,iters,threads,bx,by,rep,t_loop,t_transfer,mlups"

# run <arquivo> <setting> <comando...>: executa e acrescenta as linhas no CSV.
run() {
	local file=$1 setting=$2
	shift 2
	[ -f "$file" ] || echo "$HEADER" > "$file"
	if ! "$@" > "$tmp/out.csv" 2> "$tmp/err.txt"; then
		echo "  ERRO: $* -> $(cat "$tmp/err.txt")"
		return 1
	fi
	sed "s/^/$setting,/" "$tmp/out.csv" >> "$file"
}

# validate <N> <rótulo> <comando...>: uma execução com --dump comparada à sequencial.
validate() {
	local n=$1 label=$2
	shift 2
	local it=${ITERS[$n]}
	local vfile=$RAW/validation_runs.csv
	[ -f "$vfile" ] || echo "label,N,iters,tol,erro_abs,erro_rel,resultado" > "$vfile"
	if [ ! -f "$tmp/seq_$n.bin" ]; then
		"$BUILD/jacobi_seq" --n "$n" --iters "$it" --dump "$tmp/seq_$n.bin" > /dev/null
	fi
	local tol=1e-12
	case "$*" in *"--prec float"*) tol=1e-5 ;; esac
	"$@" --n "$n" --iters "$it" --dump "$tmp/test.bin" > /dev/null || return 1
	local out
	out=$("$BUILD/compare" "$tmp/seq_$n.bin" "$tmp/test.bin" --tol $tol)
	local code=$?
	local abs rel
	abs=$(echo "$out" | awk '/^erro_abs máx/ {print $3}')
	rel=$(echo "$out" | awk '/^erro_rel máx/ {print $3}')
	local res=OK
	[ $code -eq 0 ] || res=FALHOU
	echo "$label,$n,$it,$tol,$abs,$rel,$res" >> "$vfile"
	[ "$res" = OK ] || echo "  VALIDAÇÃO FALHOU: $label N=$n"
}

has_cuda() { [ -x "$BUILD/jacobi_cuda" ]; }

# ---------------------------------------------------------------------------
# Tempo sequencial por tamanho (referência do speedup).
exp_size() {
	echo "[size] sequencial"
	local f=$RAW/seq.csv
	rm -f "$f"
	for n in "${SIZES[@]}"; do
		run "$f" - "$BUILD/jacobi_seq" --n "$n" --iters "${ITERS[$n]}" --reps "$REPS"
	done
}

# Escalabilidade OpenMP em função das threads, para cada tamanho e variante.
exp_omp_threads() {
	echo "[omp_threads] variantes x threads x tamanhos"
	local f=$RAW/omp_threads.csv
	rm -f "$f"
	for n in "${SIZES[@]}"; do
		for t in "${THREADS[@]}"; do
			export OMP_NUM_THREADS=$t
			run "$f" - "$BUILD/jacobi_omp" --n "$n" --iters "${ITERS[$n]}" --reps "$REPS" --variant basic
			run "$f" - "$BUILD/jacobi_omp" --n "$n" --iters "${ITERS[$n]}" --reps "$REPS" --variant single
			run "$f" - "$BUILD/jacobi_omp" --n "$n" --iters "${ITERS[$n]}" --reps "$REPS" "${OMP_TEMPORAL[@]}"
		done
		export OMP_NUM_THREADS=16
		validate "$n" omp_basic_16t "$BUILD/jacobi_omp" --variant basic
		validate "$n" omp_single_16t "$BUILD/jacobi_omp" --variant single
		validate "$n" omp_temporal_16t "$BUILD/jacobi_omp" "${OMP_TEMPORAL[@]}"
		unset OMP_NUM_THREADS
	done
}

# Afinidade das threads (OMP_PROC_BIND / OMP_PLACES).
exp_affinity() {
	echo "[affinity] políticas de afinidade"
	local f=$RAW/omp_affinity.csv
	rm -f "$f"
	local settings=("none" "close:cores" "spread:cores" "close:threads")
	for n in 1024 4096; do
		[ -n "${ITERS[$n]:-}" ] || continue
		for s in "${settings[@]}"; do
			for t in 1 2 4 8 12 16; do
				if [ "$s" = none ]; then
					unset OMP_PROC_BIND OMP_PLACES
				else
					export OMP_PROC_BIND=${s%%:*} OMP_PLACES=${s##*:}
				fi
				export OMP_NUM_THREADS=$t
				run "$f" "$s" "$BUILD/jacobi_omp" --n "$n" --iters "${ITERS[$n]}" --reps "$REPS" --variant basic
				run "$f" "$s" "$BUILD/jacobi_omp" --n "$n" --iters "${ITERS[$n]}" --reps "$REPS" "${OMP_TEMPORAL[@]}"
			done
		done
	done
	unset OMP_PROC_BIND OMP_PLACES OMP_NUM_THREADS
}

# Parâmetros da blocagem temporal (lado do bloco x passos por bloco).
exp_omp_temporal() {
	echo "[omp_temporal] tile x tsteps"
	local f=$RAW/omp_temporal.csv
	rm -f "$f"
	export OMP_NUM_THREADS=8
	for n in 1024 4096; do
		[ -n "${ITERS[$n]:-}" ] || continue
		for tile in 32 64 128 256; do
			for ts in 1 2 4 8 16 32; do
				run "$f" - "$BUILD/jacobi_omp" --n "$n" --iters "${ITERS[$n]}" --reps 3 \
					--variant temporal --tile "$tile" --tsteps "$ts"
			done
		done
	done
	unset OMP_NUM_THREADS
}

# Níveis de vetorização: builds separados com flags diferentes.
exp_vectorization() {
	echo "[vectorization] sem vetorização, SSE2, AVX2"
	local f=$RAW/vectorization.csv
	rm -f "$f"
	declare -A FLAGS=([novec]="-march=native -fno-tree-vectorize -fno-tree-slp-vectorize"
	                  [sse2]="-march=x86-64 -mtune=native"
	                  [avx2]="-march=native")
	for name in novec sse2 avx2; do
		local dir=build-$name
		[ "$name" = avx2 ] && dir=$BUILD
		if [ "$name" != avx2 ]; then
			cmake -S . -B "$dir" -DJACOBI_CPU_FLAGS="${FLAGS[$name]}" > /dev/null
			cmake --build "$dir" -j --target jacobi_seq jacobi_omp > /dev/null
		fi
		for n in "${SIZES[@]}"; do
			run "$f" "$name" "$dir/jacobi_seq" --n "$n" --iters "${ITERS[$n]}" --reps 3
			OMP_NUM_THREADS=8 run "$f" "$name" "$dir/jacobi_omp" --n "$n" --iters "${ITERS[$n]}" --reps 3 --variant basic
		done
		validate 1024 "seq_$name" "$dir/jacobi_seq"
	done
}

# Dimensões do bloco CUDA para cada kernel.
exp_cuda_blocks() {
	has_cuda || return 0
	echo "[cuda_blocks] kernels x dimensões do bloco"
	local f=$RAW/cuda_blocks.csv
	rm -f "$f"
	local blocks=("8 8" "16 8" "16 16" "32 4" "32 8" "32 16" "32 32" "64 2" "64 4" "64 8" "128 1" "128 2" "256 1" "512 1" "1024 1")
	for n in 1024 4096; do
		[ -n "${ITERS[$n]:-}" ] || continue
		for blk in "${blocks[@]}"; do
			set -- $blk
			for v in basic int32 shared coarse; do
				run "$f" - "$BUILD/jacobi_cuda" --n "$n" --iters "${ITERS[$n]}" --reps 3 --variant "$v" --bx "$1" --by "$2"
			done
		done
		for blk in "32 8" "32 16" "32 32" "64 8" "64 16" "128 8"; do
			set -- $blk
			for ts in 1 2 3 4; do
				[ "$2" -gt $((2 * ts)) ] || continue
				run "$f" - "$BUILD/jacobi_cuda" --n "$n" --iters "${ITERS[$n]}" --reps 3 \
					--variant temporal --tsteps "$ts" --bx "$1" --by "$2"
			done
		done
	done
}

# Linhas por thread na variante coarse.
exp_cuda_coarsen() {
	has_cuda || return 0
	echo "[cuda_coarsen] linhas por thread"
	local f=$RAW/cuda_coarsen.csv
	rm -f "$f"
	for n in 1024 4096; do
		[ -n "${ITERS[$n]:-}" ] || continue
		for c in 1 2 4 8 16; do
			for blk in "32 4" "32 8" "64 4"; do
				set -- $blk
				run "$f" - "$BUILD/jacobi_cuda" --n "$n" --iters "${ITERS[$n]}" --reps 3 \
					--variant coarse --coarsen "$c" --bx "$1" --by "$2"
			done
		done
	done
}

# Melhor bloco de cada kernel no experimento cuda_blocks (N=4096, maior MLUPS médio).
best_block() {
	local variant=$1
	awk -F, -v v="$variant" 'NR > 1 && $5 == 4096 && $3 ~ "^"v {
		k = $8 " " $9 (v == "temporal" ? " " substr($3, 11) : ""); s[k] += $13; c[k]++
	} END { for (k in s) print s[k] / c[k], k }' "$RAW/cuda_blocks.csv" | sort -rn | head -1 | cut -d' ' -f2-
}

# Escalabilidade CUDA por tamanho: kernels, precisão, grafos e memória pinned.
exp_cuda_size() {
	has_cuda || return 0
	echo "[cuda_size] kernels x tamanhos"
	local f=$RAW/cuda_size.csv
	rm -f "$f"
	local bi bs bc bt
	bi=$(best_block int32); bs=$(best_block shared); bc=$(best_block coarse); bt=$(best_block temporal)
	bi=${bi:-32 8}; bs=${bs:-32 8}; bc=${bc:-32 8}; bt=${bt:-32 16 2}
	echo "  melhores blocos: int32=$bi shared=$bs coarse=$bc temporal=$bt"
	set -- $bi; local ibx=$1 iby=$2
	set -- $bs; local sbx=$1 sby=$2
	set -- $bc; local cbx=$1 cby=$2
	set -- $bt; local tbx=$1 tby=$2 tts=$3
	for n in "${SIZES[@]}"; do
		local it=${ITERS[$n]}
		local C=("$BUILD/jacobi_cuda" --n "$n" --iters "$it" --reps "$REPS")
		run "$f" - "${C[@]}" --variant basic --bx 16 --by 16
		run "$f" - "${C[@]}" --variant uncoalesced --bx 16 --by 16
		run "$f" - "${C[@]}" --variant int32 --bx "$ibx" --by "$iby"
		run "$f" - "${C[@]}" --variant shared --bx "$sbx" --by "$sby"
		run "$f" - "${C[@]}" --variant coarse --bx "$cbx" --by "$cby"
		run "$f" - "${C[@]}" --variant temporal --bx "$tbx" --by "$tby" --tsteps "$tts"
		run "$f" - "${C[@]}" --variant int32 --bx "$ibx" --by "$iby" --graph
		run "$f" - "${C[@]}" --variant int32 --bx "$ibx" --by "$iby" --pinned
		run "$f" - "${C[@]}" --variant int32 --bx "$ibx" --by "$iby" --graph --pinned
		run "$f" - "${C[@]}" --variant int32 --bx "$ibx" --by "$iby" --prec float
		run "$f" - "${C[@]}" --variant int32 --bx "$ibx" --by "$iby" --prec float --graph --pinned
		run "$f" - "${C[@]}" --variant temporal --bx "$tbx" --by "$tby" --tsteps "$tts" --prec float

		validate "$n" cuda_basic "$BUILD/jacobi_cuda" --variant basic --bx 16 --by 16
		validate "$n" cuda_int32 "$BUILD/jacobi_cuda" --variant int32 --bx "$ibx" --by "$iby" --graph --pinned
		validate "$n" cuda_shared "$BUILD/jacobi_cuda" --variant shared --bx "$sbx" --by "$sby"
		validate "$n" cuda_coarse "$BUILD/jacobi_cuda" --variant coarse --bx "$cbx" --by "$cby"
		validate "$n" cuda_temporal "$BUILD/jacobi_cuda" --variant temporal --bx "$tbx" --by "$tby" --tsteps "$tts"
		validate "$n" cuda_int32_float "$BUILD/jacobi_cuda" --variant int32 --bx "$ibx" --by "$iby" --prec float
	done
}

ALL=(size omp_threads affinity omp_temporal vectorization cuda_blocks cuda_coarsen cuda_size)
TODO=("$@")
[ ${#TODO[@]} -gt 0 ] || TODO=("${ALL[@]}")

[ -x "$BUILD/jacobi_seq" ] || { cmake -S . -B "$BUILD" > /dev/null && cmake --build "$BUILD" -j > /dev/null; }
scripts/collect_env.sh > /dev/null
[ " ${TODO[*]} " = " ${ALL[*]} " ] && rm -f "$RAW/validation_runs.csv"

start=$(date +%s)
for e in "${TODO[@]}"; do
	"exp_$e"
done
echo "concluído em $(($(date +%s) - start)) s. Resultados em $RAW/"

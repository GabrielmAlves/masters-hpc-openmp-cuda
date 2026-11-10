#!/usr/bin/env bash
# Valida todas as versões e variantes contra a versão sequencial.
# Grava results/validation.csv com o maior erro absoluto e relativo de cada caso.
#
# Uso: scripts/validate.sh [build_dir]
# Saída: 0 se todos os casos passarem, 1 caso contrário.

set -u
cd "$(dirname "$0")/.."

BUILD=${1:-build}
OUT=results/validation.csv
TOL_DOUBLE=1e-12 # double: o esperado é erro zero
TOL_FLOAT=1e-5   # float: precisão de ~7 dígitos (ver README)

mkdir -p results
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "version,variant,prec,N,iters,threads,bx,by,tol,erro_abs,erro_rel,identico,resultado" > "$OUT"
fails=0
total=0

# check <version> <variant> <prec> <threads> <bx> <by> <tol> <comando...>
check() {
	local version=$1 variant=$2 prec=$3 threads=$4 bx=$5 by=$6 tol=$7
	shift 7
	if ! "$@" --n "$N" --iters "$ITERS" --dump "$tmp/test.bin" > /dev/null 2> "$tmp/err.txt"; then
		echo "ERRO ao executar: $* ($(cat "$tmp/err.txt"))"
		fails=$((fails + 1))
		total=$((total + 1))
		return
	fi
	local out abs rel ident result
	out=$("$BUILD/compare" "$tmp/seq.bin" "$tmp/test.bin" --tol "$tol")
	local code=$?
	abs=$(echo "$out" | awk '/^erro_abs máx/ {print $3}')
	rel=$(echo "$out" | awk '/^erro_rel máx/ {print $3}')
	if cmp -s "$tmp/seq.bin" "$tmp/test.bin"; then ident=sim; else ident=nao; fi
	if [ $code -eq 0 ]; then result=OK; else result=FALHOU; fails=$((fails + 1)); fi
	total=$((total + 1))
	echo "$version,$variant,$prec,$N,$ITERS,$threads,$bx,$by,$tol,$abs,$rel,$ident,$result" >> "$OUT"
	if [ "$result" != OK ]; then
		echo "FALHOU: $version $variant $prec N=$N iters=$ITERS threads=$threads bloco=${bx}x$by"
	fi
}

for N in 5 64 513 1001 1024; do
	for ITERS in 1 100 101; do
		"$BUILD/jacobi_seq" --n "$N" --iters "$ITERS" --dump "$tmp/seq.bin" > /dev/null

		# OpenMP
		for t in 1 3 8 16; do
			export OMP_NUM_THREADS=$t
			check omp basic double "$t" 0 0 $TOL_DOUBLE "$BUILD/jacobi_omp" --variant basic
			check omp single double "$t" 0 0 $TOL_DOUBLE "$BUILD/jacobi_omp" --variant single
			check omp temporal_b64_t4 double "$t" 0 0 $TOL_DOUBLE \
				"$BUILD/jacobi_omp" --variant temporal --tile 64 --tsteps 4
			check omp temporal_b7_t3 double "$t" 0 0 $TOL_DOUBLE \
				"$BUILD/jacobi_omp" --variant temporal --tile 7 --tsteps 3
			check omp temporal_b128_t16 double "$t" 0 0 $TOL_DOUBLE \
				"$BUILD/jacobi_omp" --variant temporal --tile 128 --tsteps 16
		done
		unset OMP_NUM_THREADS

		# CUDA
		if [ -x "$BUILD/jacobi_cuda" ]; then
			for blk in "16 16" "32 8" "128 1" "7 5"; do
				set -- $blk
				bx=$1 by=$2
				for v in basic int32 uncoalesced shared; do
					check cuda "$v" double $((bx * by)) "$bx" "$by" $TOL_DOUBLE \
						"$BUILD/jacobi_cuda" --variant "$v" --bx "$bx" --by "$by" --reps 2
				done
				for c in 1 4 7; do
					check cuda "coarse_c$c" double $((bx * by)) "$bx" "$by" $TOL_DOUBLE \
						"$BUILD/jacobi_cuda" --variant coarse --coarsen "$c" --bx "$bx" --by "$by" --reps 2
				done
			done
			for cfg in "32 32 4" "16 16 3" "32 8 2" "64 16 7"; do
				set -- $cfg
				bx=$1 by=$2 ts=$3
				check cuda "temporal_t$ts" double $((bx * by)) "$bx" "$by" $TOL_DOUBLE \
					"$BUILD/jacobi_cuda" --variant temporal --tsteps "$ts" --bx "$bx" --by "$by" --reps 2
			done
			check cuda temporal_t4+graph double 1024 32 32 $TOL_DOUBLE \
				"$BUILD/jacobi_cuda" --variant temporal --tsteps 4 --bx 32 --by 32 --graph --reps 2
			check cuda temporal_t4 float 1024 32 32 $TOL_FLOAT \
				"$BUILD/jacobi_cuda" --variant temporal --tsteps 4 --bx 32 --by 32 --prec float --reps 2
			check cuda basic+graph double 256 16 16 $TOL_DOUBLE \
				"$BUILD/jacobi_cuda" --variant basic --graph --reps 2
			check cuda shared+graph+pinned double 256 32 8 $TOL_DOUBLE \
				"$BUILD/jacobi_cuda" --variant shared --bx 32 --by 8 --graph --pinned --reps 2
			for v in basic shared; do
				check cuda "$v" float 256 32 8 $TOL_FLOAT \
					"$BUILD/jacobi_cuda" --variant "$v" --bx 32 --by 8 --prec float --reps 2
			done
		fi
	done
done

echo "casos: $total, falhas: $fails (detalhes em $OUT)"
[ $fails -eq 0 ]

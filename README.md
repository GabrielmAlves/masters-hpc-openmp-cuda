# masters-hpc-openmp-cuda

Método iterativo de Jacobi 2D (estêncil de 5 pontos, média dos 4 vizinhos, bordas
constantes, número fixo de iterações) em três versões: sequencial (C++), OpenMP e
CUDA, com variantes otimizadas, validação numérica e scripts de experimentos.

## Requisitos

- Linux (testado no Ubuntu 24.04 sob WSL 2), GCC com OpenMP (testado com 13.3) e CMake ≥ 3.18.
- Opcional: CUDA Toolkit (testado com 13.3) e GPU NVIDIA. Sem `nvcc`, o projeto
  compila só as versões de CPU.
- Gráficos: Python 3 com matplotlib e numpy.

## Compilação

```bash
cmake -S . -B build
cmake --build build -j
```

Executáveis em `build/`: `jacobi_seq`, `jacobi_omp`, `jacobi_cuda` e `compare`, além de
`bandwidth_cpu` e `bandwidth_gpu`, que medem a banda de memória por cópia e servem de
referência para o modelo de desempenho. A análise dos resultados está em
`report/analise.md`.

Opções do CMake:

| Opção | Padrão | Uso |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `86` (RTX 3060) | Arquitetura da GPU, por exemplo `80` para A100 |
| `JACOBI_CPU_FLAGS` | `-march=native` | Flags de arquitetura da CPU (estudo de vetorização) |

Flags fixas: C++ com `-O3 -ffp-contract=off` (Release) e CUDA com `--fmad=false`. As duas
desligam a contração em FMA, para que as versões deem o mesmo resultado bit a bit.

## Execução

```bash
./build/jacobi_seq  --n 1024 --iters 1000 --reps 5 --header
OMP_NUM_THREADS=8 ./build/jacobi_omp --n 4096 --iters 60 --variant temporal --tile 128 --tsteps 16
./build/jacobi_cuda --n 4096 --iters 60 --variant int32 --bx 16 --by 8 --graph --pinned
```

Cada repetição imprime uma linha CSV:
`version,variant,prec,N,iters,threads,bx,by,rep,t_loop,t_transfer,mlups`.
`t_loop` é o tempo do laço de iterações. No CUDA, `t_transfer` é o tempo das cópias
host ↔ GPU, e `threads` é o número de threads por bloco.
`mlups` = (N−2)² × iterações / `t_loop` / 10⁶.

### Opções

| Opção | Versões | Descrição |
|---|---|---|
| `--n N` | todas | Lado da matriz, com as bordas (padrão 1024) |
| `--iters K` | todas | Número de iterações (padrão 1000) |
| `--reps R` | todas | Repetições medidas (padrão 1) |
| `--init pattern\|top` | todas | Matriz inicial (padrão `pattern`, sem zeros no interior) |
| `--dump arquivo` | todas | Grava a matriz final (binário) para o `compare` |
| `--header` | todas | Imprime o cabeçalho do CSV |
| `--variant V` | omp, cuda | Variante (tabelas abaixo) |
| `--tile B`, `--tsteps T` | omp | Blocagem temporal: lado do bloco e iterações por bloco |
| `--bx X`, `--by Y` | cuda | Dimensões do bloco de threads (padrão 16×16) |
| `--coarsen C` | cuda | Linhas por thread na variante `coarse` |
| `--tsteps T` | cuda | Iterações por lançamento na variante `temporal` |
| `--prec float\|double` | cuda | Precisão (padrão `double`) |
| `--pinned` | cuda | Memória *pinned* no host (transferências mais rápidas) |
| `--graph` | cuda | Lança as iterações com CUDA Graphs |

### Variantes OpenMP

| Variante | Descrição |
|---|---|
| `basic` | `parallel for schedule(static)` no laço das linhas, uma região paralela por iteração |
| `single` | Uma única região paralela para todas as iterações, com `omp for` dentro |
| `temporal` | Blocagem temporal com sobreposição: cada bloco `tile × tile` avança `tsteps` iterações em buffers locais da thread, que cabem na cache, e lê e grava a matriz global uma vez a cada `tsteps` iterações |

Os dois buffers são inicializados em paralelo, com a mesma divisão de linhas do cálculo (*first touch*).

### Variantes CUDA

| Variante | Descrição |
|---|---|
| `basic` | Uma thread por célula, `x` → coluna (acessos coalescidos), índices `size_t` |
| `int32` | Igual à `basic`, com índices `int` |
| `uncoalesced` | Contraexemplo: `x` → linha, acessos não coalescidos |
| `shared` | Bloco + halo em *shared memory* |
| `coarse` | Cada thread calcula `coarsen` linhas da mesma coluna, reaproveitando valores em registradores |
| `temporal` | Blocagem temporal em *shared memory*: `tsteps` iterações por lançamento |

## Validação

```bash
./build/jacobi_seq --n 1024 --iters 101 --dump seq.bin
OMP_NUM_THREADS=8 ./build/jacobi_omp --n 1024 --iters 101 --dump omp.bin
./build/compare seq.bin omp.bin            # --tol 1e-12 por padrão (erro relativo)
```

O `compare` verifica todos os elementos, informa o maior erro absoluto e o maior erro
relativo, com a posição, e detecta NaN. O código de saída é 0 (OK), 1 (fora da
tolerância) ou 2 (erro de uso).

`scripts/validate.sh` compara todas as versões e variantes com a sequencial e grava
`results/validation.csv`. Os casos combinam N = 5, 64, 513, 1001 e 1024; 1, 100 e 101
iterações; várias quantidades de threads; blocos CUDA de vários formatos, inclusive
7×5; grafos; e memória *pinned*.

- **`double`:** todas as variantes dão resultado idêntico bit a bit ao da versão
  sequencial (erro 0). A ordem das somas é a mesma em todas, e a contração em FMA
  está desligada.
- **`float`:** tolerância de `1e-5` (erro relativo). O erro relativo observado fica
  na ordem de 1e-7 a 1e-6, compatível com a precisão do `float` (cerca de 6e-8 por
  operação), acumulada ao longo das iterações.

## Experimentos

```bash
scripts/run_experiments.sh             # todos (cerca de 30 a 40 min); QUICK=1 para um teste rápido
scripts/run_experiments.sh cuda_size   # só um experimento
python3 -m venv .venv && .venv/bin/pip install matplotlib numpy
.venv/bin/python analysis/plot.py      # gráficos em results/figures/, tabelas em results/summary.md
```

| Experimento | O que mede |
|---|---|
| `size` | Tempo sequencial para N = 256 a 8192 (de 1 MB a 1 GB para as duas matrizes) |
| `omp_threads` | Speedup e eficiência das 3 variantes OpenMP com 1 a 16 threads, para cada N |
| `affinity` | `OMP_PROC_BIND` e `OMP_PLACES` |
| `omp_temporal` | Lado do bloco × iterações por bloco na blocagem temporal |
| `vectorization` | Builds sem vetorização, com SSE2 e com AVX2 |
| `cuda_blocks` | Dimensões do bloco para cada kernel CUDA |
| `cuda_coarsen` | Linhas por thread na variante `coarse` |
| `cuda_size` | Kernels × N, `float` × `double`, CUDA Graphs e memória *pinned* |

O número de iterações cai com N (15000 para N=256, até 16 para N=8192), para manter
cerca de 10⁹ atualizações por execução. Cada configuração também é conferida com o
`compare` (`results/raw/validation_runs.csv`). O hardware e o software ficam
registrados em `results/env.txt` (`scripts/collect_env.sh`).

## Estrutura

```
include/            cabeçalhos comuns (opções, matriz, CSV, cronômetro, solve)
src/common/         opções, alocação e inicialização, CSV e arquivo binário
src/seq/            versão sequencial (referência)
src/omp/            versão OpenMP (basic em solver.cpp, single e temporal em variants.cpp)
src/cuda/           versão CUDA (kernels em solver.cu, host em main.cu)
tools/compare.cpp   validação numérica
scripts/            validação, experimentos e coleta do ambiente
analysis/plot.py    gráficos e resumo
results/            CSVs, gráficos (figures/), summary.md e env.txt
```

# Jacobi 2D: análise dos resultados

Base para o relatório. Os números vêm de `results/summary.md` e os gráficos de
`results/figures/`. Os dois são gerados por `analysis/plot.py` a partir da rodada de
09/11/2026 (`scripts/run_experiments.sh`, 1041 s). Os tempos são a mediana de 5
repetições (3 nos experimentos de parâmetros), e o desempenho está em MLUPS ou GLUPS
(milhões ou bilhões de atualizações de célula por segundo).

## 1. Ambiente

| Item | Valor |
|---|---|
| CPU | Intel Core i5-13400F: 6 núcleos P (com HT) + 4 núcleos E, 16 threads; L2 de 1,25 MB por núcleo P; L3 de 20 MB |
| RAM visível no WSL | 7,7 GiB |
| GPU | NVIDIA GeForce RTX 3060 (GA106, `sm_86`): 28 SMs, 12 GB GDDR6 com barramento de 192 bits, L2 de 2,25 MB |
| Sistema | Windows 11 Pro + WSL 2 (kernel 5.15.133.1), Ubuntu 24.04.1 |
| Software | GCC 13.3.0 (`-O3 -march=native -ffp-contract=off`), CUDA 13.3 (`--fmad=false`, `sm_86`), driver 610.88, CMake 3.28.3 |

Bandas de memória medidas como referência, com programas de cópia simples:

| Memória | Medido | Teórico |
|---|---|---|
| RAM | 26 a 28 GB/s (cópia com 4 threads, 16 bytes por elemento; `build/bandwidth_cpu`) | — |
| GPU | 332 GB/s (`cudaMemcpy` e kernel de cópia; `build/bandwidth_gpu`) | 360 GB/s (7,5 GHz × 192 bits) |

**Topologia escondida pelo WSL.** O WSL mostra a CPU como 8 núcleos × 2 threads e
esconde a diferença entre núcleos P e E. Por isso, `OMP_PLACES=cores` não garante
núcleos P, e o Hyper-V pode trocar o núcleo físico de cada CPU virtual.

## 2. Modelo de desempenho

Cada atualização lê 4 vizinhos e grava 1 valor. Os vizinhos são reaproveitados entre
células próximas, então o tráfego mínimo com a memória principal é de **16 bytes por
atualização** em `double` (8 de leitura e 8 de escrita), mais 8 bytes de
*write-allocate* na CPU. São só 4 operações de ponto flutuante por atualização:
intensidade aritmética de cerca de 0,25 FLOP/byte. **O algoritmo é limitado pela banda
de memória** sempre que as matrizes não cabem na cache.

Tetos estimados a partir das bandas medidas:

| Plataforma | Teto | Conta |
|---|---|---|
| CPU com dados na RAM | ~1,7 GLUPS | 27,9 GB/s ÷ 16 B |
| GPU em `double` | ~20,8 GLUPS (22,5 com o pico teórico) | 332 GB/s ÷ 16 B |
| GPU em `float` | ~41,5 GLUPS | 332 GB/s ÷ 8 B |

Conjunto de trabalho (duas matrizes `double`) por N:

| N | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| Tamanho | 1 MB | 4 MB | 16 MB | 64 MB | 256 MB | 1 GB |
| Cabe em | L2 | L3 | L3 (por pouco) | RAM | RAM | RAM |

## 3. Validação numérica

- `scripts/validate.sh` rodou **870 casos**:
  - N = 5, 64, 513, 1001 e 1024;
  - 1, 100 e 101 iterações;
  - OpenMP com 1, 3, 8 e 16 threads e blocos temporais de vários tamanhos, inclusive 7 × 7;
  - CUDA com blocos de vários formatos, inclusive 7×5, além de grafos e memória *pinned*.
- **Todos os 825 casos em `double` dão resultado idêntico bit a bit ao da versão
  sequencial: erro absoluto e relativo iguais a 0.** O mesmo vale para as execuções
  com os tamanhos dos experimentos, até N = 8192.
  - **Por que é exato:** nenhuma variante tem redução, cada célula é calculada com a
    mesma expressão e a mesma ordem das somas (`cima + baixo + esquerda + direita`), e
    a contração em FMA está desligada (`-ffp-contract=off` e `--fmad=false`). A
    expressão nem tem uma multiplicação seguida de soma que pudesse virar FMA.
- **`float` (GPU):** o maior erro relativo observado é de **1,9e-5** (erro absoluto de
  9,9e-4), com N = 512 e 4000 iterações. O erro cresce com o número de iterações: é a
  precisão do `float` (cerca de 6e-8 por operação) acumulada ao longo da execução.
  Com 100 iterações, o erro fica em cerca de 2,5e-7.
- **Tolerâncias adotadas** (erro relativo máximo): **1e-12 em `double`** (na prática, o
  resultado é exato) e **1e-4 em `float`**. A tolerância inicial de 1e-5 em `float` foi
  ampliada porque o caso acima, sem erro de implementação, ficou fora dela: o erro
  cresce com as iterações, e 1e-4 deixa margem de cerca de 5× sobre o pior caso
  medido. Com ela, todos os casos passam.

## 4. Comparação geral

Tempo do laço e speedup sobre a versão sequencial (`results/figures/overview.png`):

| N | Sequencial | OpenMP basic (16 t) | Melhor OpenMP | CUDA basic | Melhor CUDA `double` | Melhor CUDA `float` |
|---|---|---|---|---|---|---|
| 256 | 0,219 s | 2,0× | 4,0× (`single`, 6 t) | 1,7× | 3,9× (grafos) | 9,4× |
| 512 | 0,314 s | 2,1× | 6,1× (`single`, 6 t) | 3,8× | 6,2× (grafos) | 20,3× |
| 1024 | 0,384 s | 2,4× | 4,0× (`temporal`) | 7,8× | 7,8× | 14,3× |
| 2048 | 0,908 s | 1,1× | 7,7× (`temporal`, 12 t) | 18,2× | 18,5× | 37,9× |
| 4096 | 0,862 s | 0,7× | 7,9× (`temporal`, 12 t) | 19,4× | 19,4× | 37,5× |
| 8192 | 0,938 s | 1,0× | 8,2× (`temporal`, 12 t) | 17,4× | 19,8× | 38,6× |

**A versão sequencial** cai de 4,5 GLUPS (N = 256, na L2) para 2,7 (N = 1024, no L3) e
para cerca de 1,1 GLUPS quando os dados estão na RAM. Os 1,1 GLUPS correspondem a
cerca de 26 GB/s, contando o *write-allocate*: perto da banda medida. **Uma única
thread já quase satura a memória**, e isso explica o resto da análise da CPU.

## 5. OpenMP

### 5.1 Escalabilidade da versão básica

Gráficos: `omp_speedup_threads.png` e `omp_efficiency.png`.

- **Com dados na cache** (N ≤ 1024), o speedup chega a 3,2× (N = 256), 5,5× (N = 512) e
  3,8× (N = 1024), com eficiência caindo rápido depois de 4 threads.
  - **N = 256:** são 15000 iterações curtas, de cerca de 15 µs cada. Abrir a região
    paralela e passar pela barreira a cada iteração custam caro em relação ao cálculo.
  - **N = 1024:** as matrizes ocupam quase todo o L3 compartilhado, e a banda do L3
    passa a limitar.
- **Com dados na RAM** (N ≥ 2048), o speedup é só de 1,3× a 1,4×. A melhor marca,
  cerca de 1,6 GLUPS com 4 threads, equivale a cerca de 26 GB/s com 16 bytes por
  atualização: **de 85% a 95% da banda de cópia medida**. A versão básica já está no
  teto de banda, e mais threads não trazem dados mais rápido. A cópia simples também
  atinge o máximo com 4 threads e cai depois (27,9 → 21,8 GB/s com 16 threads), o
  mesmo formato das curvas do Jacobi.
- **Acima de 12 threads, o desempenho piora**, e com 16 threads despenca em alguns casos
  (por exemplo, 0,5 GLUPS com N = 1024 e `close:cores`). Usar as 16 CPUs virtuais
  ocupa todo o processador, e o Windows e o WSL disputam os mesmos núcleos. Basta uma
  thread ser interrompida para todas esperarem na barreira, que acontece mil vezes. Os
  núcleos E, mais lentos, também atrasam as threads que caem neles, e a divisão
  estática dá a mesma quantidade de linhas a todas as threads. **Recomendação: usar no
  máximo 12 threads nesta máquina.**
- **Variabilidade:** com dados na cache, a variação entre execuções é alta no WSL. Com
  N = 1024 e 12 threads, a versão básica deu 13,5 GLUPS no experimento de afinidade e
  9,9 no de threads. Vale apresentar faixas ou citar essa variação.

### 5.2 Região paralela única (`single`)

Uma única região `parallel` para todas as iterações, em vez de uma por iteração.
- Ganha onde há muitas iterações curtas: 3,24× → 3,95× em N = 256 e 5,55× → 6,06× em
  N = 512.
- Não muda nada nos tamanhos grandes: o custo de criar a região, alguns
  microssegundos, some diante de iterações de milissegundos.
- A barreira continua lá (a do `omp for`). Só o custo de abrir e fechar a região
  desaparece.

### 5.3 Blocagem temporal (`temporal`): a otimização de maior impacto

**Ideia:** cada bloco de `tile × tile` células é copiado, com um halo de `tsteps`
células, para dois buffers locais da thread, que cabem na L2. Ali ele avança `tsteps`
iterações e só então grava o miolo de volta na matriz global. A RAM passa a ser lida e
gravada uma vez a cada `tsteps` iterações. O preço é recalcular o halo, que encolhe uma
célula por passo.

**Resultado** com dados na RAM:
- **8,8 a 9,4 GLUPS**, contra 1,1 da versão sequencial e cerca de 1,5 da básica.
- **Speedup de 7,7× a 8,2×** sobre a sequencial e **cerca de 6×** sobre a melhor
  configuração da básica.
- **Mais de 5× acima do teto de banda** de cerca de 1,7 GLUPS.
- Até com 1 thread ela é 2,4× mais rápida que a sequencial (N = 2048), porque o ganho
  vem do reaproveitamento na cache, não do paralelismo.

**Parâmetros** (`omp_temporal_params.png`, N = 4096, 8 threads):
- **Sem reuso temporal** (`tsteps = 1`): 0,8 a 1,1 GLUPS, igual à versão básica. Isso
  confirma que o ganho vem do reuso temporal, não da blocagem espacial.
- **Mais passos por bloco** dividem o tráfego por `tsteps`: o desempenho sobe até
  `tsteps` = 16 ou 32.
- **`tile` = 32:** o halo recalculado domina. Com `tsteps` = 16, a região local tem
  (32 + 32)² células para produzir 32², ou seja, 4× mais trabalho.
- **`tile` = 256:** os dois buffers locais (cerca de 1,3 MB) passam da L2 de um núcleo
  P (1,25 MB).
- **Melhor configuração:** `tile` = 128 com `tsteps` = 16 ou 32 (8,3 a 8,7 GLUPS com
  8 threads).
- Com N = 1024 (dados no L3), o melhor é `tile` = 64 com `tsteps` = 8, e o ganho sobre a
  versão básica é pequeno, porque o L3 já entrega banda suficiente.

**Escalabilidade:**
- Escala bem até 4 threads (7,3× a 7,4× sobre a sequencial com N ≥ 2048).
- Continua subindo devagar até 12 threads (7,7× a 8,2×), com uma queda em 6 threads em
  algumas medidas.
- Cai com 16 threads, pelos mesmos motivos da versão básica.
- Agora o limite é o processamento e a L2, não a RAM.

### 5.4 Afinidade

Gráfico: `omp_affinity.png`.
- Fixar as threads (`close` ou `spread`, com `cores` ou `threads`) **não teve efeito
  consistente**. Com dados na RAM, ganha até cerca de 10% com 4 threads
  (1,48 → 1,64 GLUPS). Com N = 1024, as diferenças ficam dentro da variação entre
  execuções.
- **Motivo provável:** no WSL, as threads são fixadas em CPUs virtuais, que o Hyper-V
  continua movendo entre os núcleos físicos. Em Linux nativo, o efeito esperado seria
  maior, principalmente para separar núcleos P e E.

### 5.5 Vetorização

Gráfico: `vectorization.png`.

| Situação | AVX2 / sem vetorização | SSE2 |
|---|---|---|
| Sequencial, dados na cache | 1,3× a 1,8× | no meio do caminho |
| OpenMP com 8 threads, dados na cache | 1,3× a 1,8× | no meio do caminho |
| Dados na RAM | 0,9× a 1,1× | sem diferença |

- **Com dados na RAM, a vetorização não ajuda:** quando o limite é a memória, fazer as
  contas mais rápido não muda nada.
- **Por que o ganho fica longe dos 4×** que o AVX2 permite com 4 `double` por
  instrução: o laço já é limitado pela banda da cache.
- **O resultado é idêntico bit a bit** com e sem vetorização: não há redução, então
  vetorizar não muda a ordem das operações.

## 6. CUDA

### 6.1 Versão básica e o teto de banda

- Com N ≥ 1024, o kernel básico (uma thread por célula, `x` nas colunas) alcança
  **21 a 22,6 GLUPS em `double`**.
- Isso equivale a **340 a 360 GB/s** com 16 bytes por atualização: o mesmo que a cópia
  medida (332 GB/s), dentro da variação de cerca de 5% entre execuções, e 94% a 100% do
  pico teórico.
- **O kernel básico já está no teto de banda da GPU.** Isso explica por que as
  otimizações de acesso à memória da seção 6.3 não melhoram nada.
- É cerca de **19×** a versão sequencial e **2,4×** a melhor OpenMP com dados na RAM.
  A razão entre as bandas de GPU e CPU é de 332 / 27,9 ≈ 12×.

### 6.2 Dimensões do bloco

Gráfico: `cuda_blocks.png`.
- **Blocos com pelo menos 32 threads em `x`** (64×2, 64×4, 128×2, 256×1) ficam entre 21
  e 22,6 GLUPS.
- **8×8 e 16×8** perdem de 5% a 10%: um *warp* cobre mais de uma linha e o acesso fica
  menos contínuo.
- **Blocos de 1024 threads** (32×32 e 1024×1) caem para 13 a 16 GLUPS. Um SM do
  `sm_86` comporta até 1536 threads, então só cabe um bloco de 1024 por SM, e a
  ocupação fica em 67%. Há também menos blocos para equilibrar o trabalho entre os
  28 SMs.
- A variante `coarse` sofre menos nesses blocos (17 a 19,6 GLUPS), porque cada thread
  tem mais trabalho independente.

### 6.3 Variantes de kernel

Gráfico: `cuda_size.png`, painel da esquerda.

| Variante | Resultado | Explicação |
|---|---|---|
| `int32` (índices `int` em vez de `size_t`) | Igual à `basic` com o mesmo bloco (±1% no experimento de blocos). Em `cuda_size`, a diferença em N = 8192 (19,9 contra 22,7 GLUPS) vem do bloco: a `basic` usa 16×16 e a `int32`, 64×2 | Menos instruções de endereço, mas o limite é a memória |
| `uncoalesced` (`x` nas linhas) | **2× mais lenta** (10,5 a 11,5 GLUPS) | Cada warp toca 32 linhas diferentes. Mostra a importância do acesso coalescido |
| `shared` (bloco + halo em *shared memory*) | Igual à `basic` (22,3 GLUPS) | As caches L1 e L2 já aproveitam o reuso entre vizinhos. A *shared memory* não reduz o tráfego com a memória global, que é o gargalo |
| `coarse` (4 linhas por thread) | Igual (22,1 GLUPS); 1 a 16 linhas dão quase o mesmo (`cuda_coarsen.png`) | Economiza leituras da cache, que não são o gargalo |
| `temporal` (blocagem temporal em *shared memory*) | **Resultado negativo:** 17 a 18,6 GLUPS em `double` (31 GLUPS em `float`, contra 40 da `int32`) | Ver abaixo |

**Por que a blocagem temporal não funcionou na GPU**, ao contrário da CPU:
1. **Halo recalculado.** Com 32×16 e `tsteps` = 2, só 28×12 = 336 das 512 threads
   produzem resultado (66%), e o recálculo cresce rápido com `tsteps`.
2. **Sincronização.** Há um `__syncthreads()` a cada passo, e as threads da borda ficam
   ociosas enquanto a região válida encolhe.
3. **Pouco poder de cálculo em `double`.** A GA106 faz contas em `double` a 1/64 da
   velocidade do `float` (cerca de 0,2 TFLOPS, ou cerca de 50 GLUPS no máximo com 4
   operações por atualização). O trabalho redundante consome uma fração grande disso.
   Como o `float` também perde, a sincronização e o desperdício pesam pelo menos tanto
   quanto o `double`.
4. **Pouco tráfego economizado.** A GPU já está no teto de banda, e o tráfego que sobra
   a cada lançamento (bloco com halo) ainda é grande. Uma implementação eficiente
   precisaria de blocagem 2,5D com registradores, bem mais complexa.

Mesmo com `tsteps` = 1, ela fica abaixo da `basic` (18,6 contra 22,6 GLUPS), o que mede
o custo da estrutura com dois buffers em *shared memory* e o halo carregado.

### 6.4 Custo de lançamento e CUDA Graphs

- **Matrizes pequenas:** com N = 256, cada iteração na GPU leva cerca de 10 µs. O
  tempo é dominado pelo custo de lançar o kernel, não pelo cálculo: só 6 a 7 GLUPS.
  Com N = 512, 12 a 14 GLUPS.
- **Com grafos** (100 lançamentos gravados e repetidos), N = 256 vai a **17,4 GLUPS**
  (2,8×) e N = 512 a 20,6 GLUPS (1,5×).
- **Com N ≥ 1024, os grafos não mudam nada:** o kernel é longo o bastante para esconder
  o custo de lançamento.

### 6.5 `float` × `double`

- Em `float`, a versão `int32` chega a **37 a 44 GLUPS** com N ≥ 1024: cerca de 2× a
  versão em `double`, porque move metade dos bytes. Isso bate com o teto de cerca de
  41,5 GLUPS e é outra confirmação de que o limite é a banda de memória, não o
  processamento em `double`.
- **N = 512 em `float` chega a 67 GLUPS:** as duas matrizes somam 2 MB e cabem na L2 de
  2,25 MB, então o gargalo deixa de ser a memória principal.
- O preço é a precisão: erro relativo de até 1,9e-5 (seção 3).

### 6.6 Transferências entre o host e a GPU

Gráfico: `cuda_transfer.png`. São 2 envios e 1 retorno da matriz.

| | N = 4096 (3 × 128 MB) | Banda |
|---|---|---|
| Memória paginável | 54 ms | 7,1 GB/s |
| Memória *pinned* | 16 ms | 23,6 GB/s (compatível com PCIe 4.0 x16) |

- **Peso no total** (com a paginável): 55% com N = 4096 (60 iterações) e 80% com
  N = 8192 (16 iterações). Com *pinned*, cai para 25% e 58%.
- **Speedup de ponta a ponta** com N = 8192, contando as transferências: 4,1×
  (paginável) e 8,2× (*pinned*), contra 19,8× contando só o laço.
- **Esse peso depende do número de iterações:** as transferências acontecem uma vez,
  e o custo do laço cresce com as iterações. Para executar até convergir, com milhares
  de iterações, elas ficam desprezíveis.

## 7. Gargalos e conclusões

1. **O gargalo é a banda de memória, nas duas plataformas.** O Jacobi faz cerca de
   0,25 FLOP por byte. As versões diretas (OpenMP básico e CUDA básico) já operam
   perto do teto de banda medido: cerca de 90% na CPU e cerca de 100% na GPU.
2. **Na CPU,** paralelizar sem reduzir o tráfego dá no máximo 1,4× com dados na RAM.
   **A blocagem temporal**, que reaproveita os dados na cache por várias iterações, foi
   a otimização decisiva: cerca de 8× sobre a sequencial e cerca de 6× sobre a
   paralelização direta.
3. **Na GPU,** a versão básica com acesso coalescido já fica no teto. As otimizações
   clássicas (*shared memory*, *coarsening*, índices `int`) não melhoram nada, e a
   blocagem temporal ingênua piora. O que funcionou:
   - **reduzir os bytes**, com `float` (2×);
   - **reduzir o custo de lançamento** com grafos, nas matrizes pequenas (até 2,8×);
   - **reduzir o custo das transferências** com memória *pinned* (3,3×).
   - O contraexemplo não coalescido mostra o custo de errar o padrão de acesso (2×).
4. **No ambiente,** o processador híbrido sob o WSL limita a escalabilidade acima de
   12 threads e aumenta a variação entre execuções. A afinidade tem pouco efeito,
   porque o WSL não expõe a topologia real. Esses fatores devem ser citados como
   limitação dos experimentos.

## 8. Limitações e trabalhos futuros

- Rodar em Linux nativo (ou numa máquina da disciplina) para separar núcleos P e E e
  medir a afinidade de verdade.
- Blocagem temporal na GPU com uma abordagem 2,5D, com *streaming* ao longo de uma
  dimensão e registradores, que costuma superar a versão ingênua.
- Comparar com uma GPU de datacenter, mais rápida em `double`, para separar o efeito do
  `double` do efeito da sincronização na blocagem temporal.
- Medir a energia consumida e usar um *profiler* (Nsight Compute) para confirmar a banda
  efetiva dos kernels.

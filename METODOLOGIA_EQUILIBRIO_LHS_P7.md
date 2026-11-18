# Metodologia de Equilíbrio Estatístico e Substituição *Space-Filling* — Família $P_7$ (LHS 6-DoF)

Este documento descreve rigorosamente as métricas estatísticas e o algoritmo de substituição sequencial implementados em [`lhs_diagnostics.py`](file:///C:/Users/phavm/Documents/dev/python/visual-servoing-dataset/src/visual_servoing_dataset/robot/lhs_diagnostics.py) e [`pose_generator.py`](file:///C:/Users/phavm/Documents/dev/python/visual-servoing-dataset/src/visual_servoing_dataset/robot/pose_generator.py) para garantir o equilíbrio experimental do conjunto de poses combinadas **$P_7$ (6-DoF)** no protocolo *Visual Servoing Dataset Collector*.

---

## 1. Motivação em Projeto de Experimentos (DoE)

A família **$P_7$** amostra $N = 30$ poses simultaneamente nos 6 graus de liberdade locais $\xi = [\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z] \in SE(3)$, mapeadas a partir do hipercubo unitário normalizado $U \in [0, 1]^{N \times 6}$ para os limites efetivos $[l_{\text{eff}}, u_{\text{eff}}] = \alpha \cdot [l, u]$ (com $\alpha = 0{,}75$):

$$\xi_{i,j} = l_{\text{eff}, j} + u_{i,j} \cdot (u_{\text{eff}, j} - l_{\text{eff}, j}), \qquad j \in \{x, y, z, r_x, r_y, r_z\}$$

### Por que auditar tanto o conjunto original quanto as substituições?
1. **Conjunto Original (`generate_plan`)**: Gerado com `scipy.stats.qmc.LatinHypercube(d=6, optimization="random-cd")`, que impõe estratificação exata ($1$ amostra em cada um dos $N$ estratos de cada eixo) e minimiza a discrepância centrada ($CD_2$). Ainda assim, é fundamental medir se as projeções marginais, a correlação cruzada entre eixos e o preenchimento 6D satisfazem critérios formais para $N=30, d=6$.
2. **Conjunto Pós-Substituição (`verify_plan`)**: Durante a verificação física com o robô e o ROI vermelho de 80% no [`stream.py`](file:///C:/Users/phavm/Documents/dev/python/visual-servoing-dataset/stream.py), alguma amostra $u_k$ de $P_7$ pode exigir rejeição (por sair do campo de visão ou atingir limite cinemático `OutRange`).
   - Uma **substituição ingênua** (`LatinHypercube(d=6).random(n=1)`) degenera para um único ponto aleatório uniforme independente, ignorando as $N-1$ poses já aprovadas — o que pode criar aglomerados, deixar estratos vazios e introduzir correlação espuria ou viés de distribuição.
   - A **substituição *Space-Filling* equilibrada** implementada aqui otimiza a amostra substituta em função de todo o conjunto ativo $U_{\text{ativo}}$ enquanto evita a vizinhança da região física que falhou em $U_{\text{rej}}$.

---

## 2. Métricas Estatísticas Implementadas

Todas as métricas são avaliadas sobre as coordenadas normalizadas $U \in [0, 1]^{N \times 6}$ por [`evaluate_lhs_sample_set`](file:///C:/Users/phavm/Documents/dev/python/visual-servoing-dataset/src/visual_servoing_dataset/robot/lhs_diagnostics.py#L84-L280) e [`compare_lhs_original_vs_final`](file:///C:/Users/phavm/Documents/dev/python/visual-servoing-dataset/src/visual_servoing_dataset/robot/lhs_diagnostics.py#L283-L374).

### 2.1. Uniformidade Global 6D (Discrepância e Referência Monte Carlo)
- **Centered $L_2$-Discrepancy ($CD_2$)** e **Wrap-Around $L_2$-Discrepancy ($WD_2$)** (`scipy.stats.qmc.discrepancy`): medem o desvio da distribuição empírica em relação à distribuição uniforme contínua em $[0, 1]^6$. Valores menores indicam melhor preenchimento uniforme do volume 6D.
- **Calibração contra *Baseline* Monte Carlo (`trials = 500`, `seed = 20260101`)**: como o valor bruto de $CD_2$ depende da dimensão $d=6$ e do tamanho amostral $N=30$, o módulo compara $CD_2(U)$ contra 500 realizações independentes de amostragem aleatória pura $\mathcal{U}([0,1]^6)$ de mesmo $(N, d)$, reportando o **Percentil Monte Carlo** (`cd2_percentile_vs_mc`) e o **$z$-score** (`cd2_zscore_vs_mc`).

### 2.2. Uniformidade Marginal, Estratificação e Balanço de Sinais (por Eixo)
Para cada um dos 6 eixos $j \in \{x, y, z, r_x, r_y, r_z\}$:
- **Teste de Kolmogorov-Smirnov 1D (`scipy.stats.kstest`)**: testa a hipótese nula $H_0: u_{\cdot,j} \sim \mathcal{U}(0, 1)$, reportando a estatística $D_{\text{KS}}$ e o $p$-valor.
- **Centralização e Momentos**: verifica a média amostral $\bar{u}_j$ (ideal $= 0{,}50$), o desvio $|\bar{u}_j - 0{,}50|$ e o desvio-padrão $s_j$ frente ao valor teórico $\sigma_{\mathcal{U}(0,1)} = 1/\sqrt{12} \approx 0{,}2887$.
- **Balanço de Sinais ($-/+$ em torno de $P_0$)**: conta amostras com perturbação negativa ($u_{i,j} < 0{,}5$) vs. positiva ($u_{i,j} \ge 0{,}5$) e aplica teste binomial exato (`scipy.stats.binomtest` sob $p=0{,}5$).
- **Cobertura de Estratos 1D**: divide $[0, 1)$ em $N$ intervalos $[b/N, (b+1)/N)$ e mede a fração $\text{estratos únicos} / N$ ($100\%$ no LHS original).

### 2.3. Ortogonalidade e Condicionamento entre Eixos
- **Correlações Lineares e Monotônicas**: matriz $6 \times 6$ de **Pearson** e **Spearman** (`scipy.stats.spearmanr`), extraindo $\rho_{\max} = \max_{j < k} |r_{jk}|$, a correlação quadrática média ($\rho_{\text{RMS}}$) e o pior par de eixos (`worst_pearson_pair`).
- **Número de Condicionamento ($\kappa$)**: calculado sobre a matriz padronizada (média zero, variância unitária) para detectar eventual colinearidade multivariada em regressões/análises de sensibilidade futuras.

### 2.4. Espaçamento Geométrico e Projeções em Subespaços
- **Critério *Maximin* ($d_{\min}$)**: menor distância Euclidiana entre qualquer par de amostras em $[0, 1]^6$ (`scipy.spatial.distance.pdist`), garantindo ausência de pontos quase duplicados.
- **Critério Morris-Mitchell ($\phi_p$, com $p = 10$)**:
  $$\phi_{10}(U) = \left( \sum_{1 \le i < k \le N} \|u_i - u_k\|_2^{-10} \right)^{1/10}$$
- **Critério *MaxPro* (Joseph et al., 2015)**: avalia simultaneamente a qualidade do espaçamento em todas as projeções de subespaços (1D, 2D, 3D, ...):
  $$\text{MaxPro}(U) = \left( \frac{1}{\binom{N}{2}} \sum_{1 \le i < k \le N} \prod_{j=1}^{d} \frac{1}{(u_{ij} - u_{kj})^2} \right)^{1/d}$$

### 2.5. Auditoria de Viés Pós-Substituição (Original vs. Conjunto Final)
Quando há substituições ($U_{\text{final}} \neq U_{\text{orig}}$), [`compare_lhs_original_vs_final`](file:///C:/Users/phavm/Documents/dev/python/visual-servoing-dataset/src/visual_servoing_dataset/robot/lhs_diagnostics.py#L283-L374) confronta o conjunto original com o conjunto final aprovado:
- **Teste KS de Duas Amostras (`scipy.stats.ks_2samp`)** por eixo ($H_0$: mesma distribuição marginal antes e depois das rejeições);
- **Distância de Wasserstein 1D (`scipy.stats.wasserstein_distance`)** por eixo, medindo o deslocamento de massa de probabilidade;
- **Variações Globais**: $\Delta CD_2$ (absoluto e percentual), $\Delta \rho_{\max}$, $\Delta d_{\min}$, razão $d_{\min,\text{final}} / d_{\min,\text{orig}}$ e $\Delta \text{MaxPro}$.

---

## 3. Algoritmo de Substituição Equilibrada (*Space-Filling Replacement*)

Implementado em [`select_optimal_lhs_replacement`](file:///C:/Users/phavm/Documents/dev/python/visual-servoing-dataset/src/visual_servoing_dataset/robot/lhs_diagnostics.py#L377-L484) e invocado por [`generate_lhs_replacement_candidate`](file:///C:/Users/phavm/Documents/dev/python/visual-servoing-dataset/src/visual_servoing_dataset/robot/pose_generator.py#L200-L254):

1. **Identificação dos Estratos Vazios**: dado o conjunto ativo atual $U_{\text{ativo}}$ e o tamanho alvo $N$, identifica em cada eixo $j \in \{0 \dots 5\}$ quais estratos ainda estão desocupados após a remoção das poses rejeitadas.
2. **Geração de Pool Híbrido Determinístico** (`pool_size = 512`, com semente determinística `replacement_seed = base_seed + 1000 + replacement_index`):
   - **50% guiados por estratos vazios** (metade estritamente nos estratos livres e metade permitindo $\pm 1$ estrato adjacente para escapar de regiões de borda);
   - **50% via LHS uniforme em $[0, 1]^6$** (`qmc.LatinHypercube`).
3. **Exclusão da Região Rejeitada**: descarta candidatos cuja distância Euclidiana até qualquer pose em $U_{\text{rej}}$ seja inferior a `exclusion_radius = 0.15` (com *fallback* determinístico para os 25% mais distantes caso o pool fique restrito), evitando insistir na vizinhança da pose inviável.
4. **Seleção Multi-Critério sobre $U_{\text{ativo}} \cup \{u_i\}$**: após pré-filtragem vetorizada dos 64 melhores finalistas, avalia a união exata $U_{\text{ativo}} \cup \{u_i\}$ e escolhe a amostra $u^*$ que minimiza o custo normalizado:
   $$J(u_i) = 0{,}45\,\widetilde{CD_2} + 0{,}25\,\widetilde{\rho_{\max}} + 0{,}20\,(1 - \widetilde{d_{\min}}) + 0{,}10\,(1 - \text{taxa\_estratos\_vazios})$$

---

## 4. Critérios Numéricos de Aceitação (`DEFAULT_BALANCE_THRESHOLDS`)

| Verificação | Limiar Padrão | Significado Prático |
| :--- | :---: | :--- |
| `sample_size_sufficient` | $N \ge 6$ | Mínimo para inferência estatística em 6-DoF |
| `space_filling_cd2_ok` | Percentil MC $\le \mathbf{15{,}0\%}$ | $CD_2$ significativamente melhor que amostragem aleatória pura |
| `ks_uniformity_ok` | $\min_j p_{\text{KS},j} \ge \mathbf{0{,}05}$ | Nenhuma margem desviou significativamente de $\mathcal{U}(0,1)$ |
| `stratum_coverage_ok` | Cobertura mín. $\ge \mathbf{0{,}85}$ | Ao menos $85\%$ dos estratos 1D ocupados em todos os eixos |
| `mean_centering_ok` | $\max_j |\bar{u}_j - 0{,}5| \le \mathbf{0{,}08}$ | Centro amostral alinhado a $P_0$ em todos os eixos |
| `orthogonality_ok` | $\max |r| \le \mathbf{0{,}30}$ e $\text{RMS} \le \mathbf{0{,}16}$ | Baixa correlação linear entre os 15 pares de DoFs |
| `no_distribution_shift_ok` *(pós-subst.)* | $\min_j p_{\text{KS-2samp},j} \ge \mathbf{0{,}05}$ | As substituições não alteraram a distribuição marginal original |
| `min_dist_preserved_ok` *(pós-subst.)* | $d_{\min,\text{final}}/d_{\min,\text{orig}} \ge \mathbf{0{,}70}$ | Nenhuma amostra substituta colapsou sobre vizinhos existentes |

---

## 5. Uso via CLI e Artefatos Registrados

1. **Na geração do plano (Fase 2A)**:
   ```bash
   poetry run python main.py --generate-plan
   ```
   Exibe a tabela Rich do diagnóstico inicial para cada cena, grava `lhs_balance_report` dentro de `assets/dataset/poses_plan.json` e exporta `assets/dataset/lhs_balance_report.md`.

2. **Na verificação supervisionada com substituições (Fase 2B)**:
   ```bash
   poetry run python main.py --verify-plan --interactive
   ```
   Cada pose rejeitada em $P_7$ aciona [`select_optimal_lhs_replacement`](file:///C:/Users/phavm/Documents/dev/python/visual-servoing-dataset/src/visual_servoing_dataset/robot/lhs_diagnostics.py#L430-L535), salva os metadados de escolha em `replacement_selection_metrics` e atualiza o comparativo **Original vs. Final** antes do congelamento SHA-256 em `poses_plan.json` e em `lhs_balance_report.md`.

   Caso o `poses_plan.json` já esteja com `status: "frozen"` e você precise corrigir ou reavaliar poses específicas sem descartar o restante do trabalho:
   ```bash
   # Reavaliar interativamente 2 poses específicas de uma cena (ex.: scene_01)
   poetry run python main.py --verify-plan --interactive --scenes scene_01 --reverify-poses p7_lhs_005 p7_lhs_012

   # Ou rejeitar e substituir diretamente poses específicas via CLI (usando prefixo cena:pose_id ou --scenes)
   poetry run python main.py --verify-plan --interactive --scenes scene_01 --reject-poses p7_lhs_005 p7_lhs_012

   # Ou reavaliar todas as poses P7 já aprovadas de uma cena (preservando histórico de rejeições)
   poetry run python main.py --verify-plan --interactive --scenes scene_01 --reverify-all
   ```
   *(Toda reabertura preserva integralmente as poses já marcadas como `rejected`, evita colisão nos índices `_repXX` / `replacement_seed`, atualiza `lhs_balance_report` e recalcula a assinatura `plan_sha256`).*

3. **Auditoria independente a qualquer momento**:
   ```bash
   poetry run python main.py --analyze-plan --plan assets/dataset/poses_plan.json
   ```
   Reavalia todas as cenas de um plano existente, imprime o resumo no terminal e regenera `lhs_balance_report.md`.

4. **Persistência no Dataset Final (Fase 3)**:
   Ao executar `--collect`, o [`dataset_collector.py`](file:///C:/Users/phavm/Documents/dev/python/visual-servoing-dataset/src/visual_servoing_dataset/collector/dataset_collector.py#L513-L530) copia integralmente `lhs_balance_report` para `metadata.json` em cada cena (`scenes[scene_id].lhs_balance_report`), garantindo rastreabilidade científica completa na dissertação.

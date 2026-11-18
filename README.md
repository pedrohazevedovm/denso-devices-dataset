# Visual Servoing Dataset Collector (DENSO VS-050 + Android Cam-App)

Sistema em Python para automação da calibração câmera/Hand-Eye e captura de datasets de visão computacional e *Visual Servoing* com rigor cinemático em $SE(3)$, amostragem via **Latin Hypercube Sampling (LHS)** estatisticamente auditada e controle de qualidade estrito. O sistema controla o robô industrial **DENSO VS-050** (via `aether_rdk`) e sincroniza disparos fotográficos pareados de smartphones Android utilizando exclusivamente o aplicativo dedicado **`Cam-App`** (servidor HTTP Camera2 HAL na porta 8080).

---

## 🌟 Principais Funcionalidades

- **Fase 0 — Calibração Automatizada "2 em 1" via ChArUco (`calibration/`)**:
  - **Centralização e Paralelismo Automático em $P_{\text{cal},0}$ (`--center-charuco`)**: Malha fechada visual usando o marcador central **`ID = 8`** (com escala métrica $\text{mm/px}$ direta) e equalização de perímetros das extremidades (`15, 16` vs `0, 1` e Topo vs Base) para garantir centralização e paralelismo mesmo ao trocar o smartphone no suporte.
  - **Abóbada Esférica Centrada no Alvo (`calibration_generator.py`)**: Geração de 24 poses de calibração em $SE(3)$ orbitando o pivô virtual no centro do tabuleiro ChArUco (${}^B T_{\text{pivot}}$), mantendo $+Z_T$ apontado para o centro do tabuleiro em 3 blocos de 8 poses (Anel Interno de desacoplamento focal, Anel Externo de excitação Hand-Eye 6D e Varredura de Borda/Canto para distorção radial/tangencial).
  - **Solucionador Conjunto (`calibration_solver.py`)**: Detecção subpixel de cantos (`cv2.aruco.CharucoDetector`), calibração intrínseca ($K, D$ via `cv2.calibrateCamera` com chute inicial opcional da API `Camera2`), estimação de pose por vista (`cv2.solvePnP`), calibração **Hand-Eye** ($AX = XB \Rightarrow {}^T T_C$ via `cv2.calibrateHandEye`) e validação cruzada da invariância da pose do tabuleiro na base (${}^B T_M$).
  - **Retificação por LUT e Homografia Planar Ground-Truth**: Retificação rápida (`ImageRectifier` via `cv2.initUndistortRectifyMap` + `cv2.remap`) e cálculo da homografia teórica induzida pelo plano ($H_{\text{gt}} = K_{\text{rect}} (R_{\text{rel}} - \frac{t_{\text{rel}} n_0^T}{d_0}) K_{\text{rect}}^{-1}$).

- **Cinemática Rigorosa em $SE(3)$ (`kinematics.py`)**:
  - Proibição de soma direta de offsets de Euler no referencial da base.
  - Conversão bidirecional entre objetos `Pose(X, Y, Z, Rx, Ry, Rz, Fig)` do DENSO RC8 e matrizes homogêneas $4 \times 4 \in SE(3)$ (convenção configurável no `config.yaml`, padrão `xyz` extrínseco / `ZYX` intrínseco).
  - Pós-multiplicação no referencial local do TCP inicial $T_0$:
    $${}^B T_{T,i}^{\text{alvo}} = {}^B T_T(0) \cdot \Delta T_{T,i}$$
  - Cálculo exato da transformação relativa efetivamente executada e da pose real do centro óptico da câmera (${}^B T_{C,k} = {}^B T_{T,k} \cdot {}^T T_C$).

- **Famílias de Poses $P_0 \dots P_7$ com Limites por Cena e Equilíbrio Estatístico Auditado (`pose_generator.py`, `lhs_diagnostics.py`)**:
  - **Limites `[lower_bounds, upper_bounds]` por Cena**: Configurados individualmente dentro de cada item de `scenes:` em [`config.yaml`](file:///C:/Users/phavm/Documents/dev/python/visual-servoing-dataset/config.yaml) e validados por `SceneConfig.resolve_bounds()`.
  - **$P_0$ (`p0_reference_start` e `p0_reference_end`)**: Pose de referência ($\xi = \mathbf{0}$) capturada no início e obrigatoriamente recapturada ao final do bloco para verificação de deriva mecânica e fotométrica.
  - **$P_1 \dots P_6$ (Perturbações Isoladas Determinísticas)**: Translações e rotações puras no referencial $T_0$, amostradas em frações configuráveis dos limites $[l, u]$.
  - **$P_7$ (Perturbações Combinadas em 6-DoF via LHS)**: Geradas via `scipy.stats.qmc.LatinHypercube(d=6, optimization="random-cd", seed=seed)` ($N = 30$ amostras padrão) com fator de encolhimento de segurança `lhs_shrinkage_factor` (`0.75`).
  - **Substituição Equilibrada (*Space-Filling*) e Auditoria Estatística (`lhs_diagnostics.py`)**: Em caso de rejeição durante `--verify-plan`, amostras substitutas são escolhidas por otimização multi-critério (*Min-$CD_2$*, *Maximin*, baixa correlação e estratos vazios, com raio de exclusão da região rejeitada). O equilíbrio **Original vs. Final** ($CD_2$ vs. Monte Carlo, KS 1D e 2-amostras, Wasserstein 1D, Pearson/Spearman, *MaxPro* e cobertura de estratos) é salvo em `poses_plan.json`, `lhs_balance_report.md` e `metadata.json`. Consulte a documentação detalhada em [`METODOLOGIA_EQUILIBRIO_LHS_P7.md`](file:///C:/Users/phavm/Documents/dev/python/visual-servoing-dataset/METODOLOGIA_EQUILIBRIO_LHS_P7.md).

- **Visualizador ao Vivo em Tempo Real (`stream.py`)**:
  - Live stream HTTP estilo `scrcpy` operando exclusivamente em modo `auto`, com grade de enquadramento, mira central, retângulo pontilhado vermelho delimitando o **ROI 80%** (`0.8 x 0.8`), HUD de telemetria (FPS, ISO, exposição, foco em Dioptrias/cm) e detector ChArUco ao vivo (`--charuco`).

- **Duas Configurações Pareadas no `Cam-App` (`auto` e `manual_calibrado`)**:
  - **`auto`**: Rotinas 3A do Android (AF, AE, AWB), pós-processamento ISP e estabilização óptica/eletrônica (OIS/EIS) ativados.
  - **`manual_calibrado`**: Exposição, sensibilidade, foco e balanço de branco fixos, com controles de pós-processamento e estabilização desativáveis configurados para OFF. Calibrado automaticamente a partir da leitura do `X-Camera-Metadata` em modo `auto` na pose inicial $P_0$.

---

## 📌 Estrutura do Projeto

```text
visual-servoing-dataset/
├── pyproject.toml                     # Dependências Poetry (aether-rdk, opencv-python, scipy, numpy)
├── config.yaml                        # Configuração global (ChArUco, Abóbada, limites por cena, LHS)
├── main.py                            # CLI do protocolo experimental (Fase 0 + Fases 1 a 3 + --analyze-plan)
├── stream.py                          # Visualizador ao vivo (modo auto + ROI pontilhado 80% + ChArUco)
├── RELATORIO_CALIBRACAO_FASE0.md      # Relatório técnico da Fase 0 (Intrínsecos + Hand-Eye)
├── METODOLOGIA_EQUILIBRIO_LHS_P7.md   # Metodologia estatística e substituição Space-Filling da família P7
├── src/
│   └── visual_servoing_dataset/
│       ├── config.py                  # Dataclasses de configuração (validação e resolve_bounds por cena)
│       ├── schemas.py                 # Esquemas de planos (SHA-256), calibração, lhs_balance_report e metadados
│       ├── calibration/
│       │   ├── calibration_generator.py # Geração do ChArUco PNG/JSON e Abóbada Esférica (24 poses)
│       │   ├── calibration_solver.py    # CharucoDetector, calibrateCamera, calibrateHandEye, Rectifier, H_gt
│       │   └── calibration_manager.py   # Orquestração da Fase 0, auto_center_charuco e calibration_profile.json
│       ├── robot/
│       │   ├── denso_controller.py  # Wrapper da aether-rdk (controle cartesiano, juntas e OutRange seguro)
│       │   ├── kinematics.py        # Conversões Pose <-> SE(3) 4x4, pós-multiplicação e inversa analítica
│       │   ├── lhs_diagnostics.py   # Métricas estatísticas LHS (CD2, KS, MaxPro) e substituição Space-Filling
│       │   ├── pose_generator.py    # Gerador das famílias P0..P7 e Latin Hypercube Sampling (P7)
│       │   └── trajectories.py      # Adaptador de compatibilidade
│       ├── smartphone/
│       │   └── phone_controller.py  # Cliente HTTP Cam-App, extração X-Camera-Metadata e calibração P0
│       ├── collector/
│       │   ├── session_manager.py   # Fases 1 e 2 (explore-limits, generate-plan, verify-plan, analyze-plan)
│       │   └── dataset_collector.py # Fase 3 (coleta pareada + retificação + H_gt + QA + CSV/JSON)
│       └── utils/
│           └── logger.py            # Logger formatado com Rich
└── tests/
    ├── test_protocol_se3_lhs.py     # Testes unitários das Fases 1-3 (SE(3), P0..P7, limites por cena)
    ├── test_lhs_balance.py          # Testes estatísticos de equilíbrio LHS e substituição Space-Filling
    └── test_calibration_pipeline.py # Testes unitários e sintéticos da Fase 0 (Abóbada, Auto-Center, K/D/Hand-Eye)
```

---

## 🚀 Fluxo Operacional Completo (Fase 0 + Fases 1 a 3)

### Fase 0: Calibração Automatizada "2 em 1" (Intrínsecos + Hand-Eye via ChArUco)
1. **Gerar prancha ChArUco imprimível (PNG em alta resolução + `charuco_spec.json` para impressão 1:1)**:
   ```bash
   poetry run python main.py --generate-charuco
   ```
2. **Centralizar automaticamente (opcional isolado) e executar a calibração de 24 poses na Abóbada Esférica**:
   ```bash
   poetry run python main.py --center-charuco
   poetry run python main.py --calibrate-camera --verify-poses --interactive
   ```
   Gera `assets/dataset/calibrations/<calibration_id>/calibration_profile.json` contendo $K$, $D$, $K_{\text{rect}}$, ${}^T T_C$ (Hand-Eye) e ${}^B T_{\text{board}}$.

### Live View (`stream.py`) e Fase 1: Exploração de Limites (`--explore-limits`)
Abra o stream em um terminal para visualizar o enquadramento com o **ROI pontilhado vermelho de 80%**:
```bash
poetry run python stream.py
```
E explore interativamente os limites `[lower_bounds, upper_bounds]` de cada cena:
```bash
poetry run python main.py --explore-limits --interactive
```

### Fase 2: Geração, Verificação Supervisionada e Auditoria Estatística (`--generate-plan`, `--verify-plan`, `--analyze-plan`)
```bash
# 2A: Gerar plano candidato (P0..P7 + relatório inicial de equilíbrio P7)
poetry run python main.py --generate-plan

# 2B: Verificar poses P7 com substituição Space-Filling de rejeições e congelar com SHA-256
poetry run python main.py --verify-plan --interactive

# Reavaliar ou rejeitar poses específicas em um plano já congelado (preservando histórico)
poetry run python main.py --verify-plan --interactive --scenes scene_01 --reverify-poses p7_lhs_005 p7_lhs_012
poetry run python main.py --verify-plan --interactive --scenes scene_01 --reject-poses p7_lhs_005 p7_lhs_012
poetry run python main.py --verify-plan --interactive --scenes scene_01 --reverify-all

# Auditar equilíbrio estatístico Original vs. Final a qualquer momento (gera lhs_balance_report.md)
poetry run python main.py --analyze-plan --plan assets/dataset/poses_plan.json
```

### Fase 3: Execução da Coleta Automática Pareada (`--collect`) e Retificação (`--rectify-dataset`)
```bash
# Coleta pareada (injeta calibration_profile.json e lhs_balance_report no metadata.json e salva imagens retificadas)
poetry run python main.py --collect --plan assets/dataset/poses_plan.json

# Retificar retroativamente um dataset existente
poetry run python main.py --rectify-dataset --calibration-file assets/dataset/calibrations/calib_motog2025_mount01_20261113/calibration_profile.json
```

### Simulação Completa (`--dry-run`) e Testes Unitários (`pytest`)
```bash
# Testar Fase 0 + Fases 1 a 3 em modo de simulação
poetry run python main.py --generate-charuco --calibrate-camera --explore-limits --generate-plan --verify-plan --collect --dry-run

# Executar toda a suíte de testes unitários e estatísticos
poetry run pytest -v
```

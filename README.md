# Visual Servoing Dataset Collector (DENSO VS-050 + Android Cam-App)

Sistema em Python para automação da calibração câmera/Hand-Eye e captura de datasets de visão computacional e *Visual Servoing* com rigor cinemático em $SE(3)$, amostragem via **Latin Hypercube Sampling (LHS)** e controle de qualidade estrito. O sistema controla o robô industrial **DENSO VS-050** (via `aether_rdk`) e sincroniza disparos fotográficos pareados de smartphones Android utilizando exclusivamente o aplicativo dedicado **`Cam-App`** (servidor HTTP Camera2 HAL na porta 8080).

---

## 🌟 Principais Funcionalidades

- **Fase 0 — Calibração Automatizada "2 em 1" via ChArUco (`calibration/`)**:
  - **Abóbada Esférica Centrada no Alvo (`calibration_generator.py`)**: Geração de 24 poses de calibração em $SE(3)$ orbitando o pivô virtual no centro do tabuleiro ChArUco (${}^B T_{\text{pivot}}$), mantendo $+Z_T$ apontado para o centro do tabuleiro em 3 blocos de 8 poses (Anel Interno de desacoplamento focal, Anel Externo de excitação Hand-Eye 6D e Varredura de Borda/Canto para distorção radial/tangencial).
  - **Solucionador Conjunto (`calibration_solver.py`)**: Detecção subpixel de cantos (`cv2.aruco.CharucoDetector`), calibração intrínseca ($K, D$ via `cv2.calibrateCamera` com chute inicial opcional da API `Camera2`), estimação de pose por vista (`cv2.solvePnP`), calibração **Hand-Eye** ($AX = XB \Rightarrow {}^T T_C$ via `cv2.calibrateHandEye`) e validação cruzada da invariância da pose do tabuleiro na base (${}^B T_M$).
  - **Retificação por LUT e Homografia Planar Ground-Truth**: Retificação rápida (`ImageRectifier` via `cv2.initUndistortRectifyMap` + `cv2.remap`) e cálculo da homografia teórica induzida pelo plano ($H_{\text{gt}} = K_{\text{rect}} (R_{\text{rel}} - \frac{t_{\text{rel}} n_0^T}{d_0}) K_{\text{rect}}^{-1}$).

- **Cinemática Rigorosa em $SE(3)$ (`kinematics.py`)**:
  - Proibição de soma direta de offsets de Euler no referencial da base.
  - Conversão bidirecional entre objetos `Pose(X, Y, Z, Rx, Ry, Rz, Fig)` do DENSO RC8 e matrizes homogêneas $4 \times 4 \in SE(3)$ (convenção configurável no `config.yaml`, padrão `xyz` extrínseco / `ZYX` intrínseco).
  - Pós-multiplicação no referencial local do TCP inicial $T_0$:
    $${}^B T_{T,i}^{\text{alvo}} = {}^B T_T(0) \cdot \Delta T_{T,i}$$
  - Cálculo exato da transformação relativa efetivamente executada e da pose real do centro óptico da câmera (${}^B T_{C,k} = {}^B T_{T,k} \cdot {}^T T_C$).

- **Famílias de Poses $P_0 \dots P_7$ (`pose_generator.py`)**:
  - **$P_0$ (`p0_reference_start` e `p0_reference_end`)**: Pose de referência ($\xi = \mathbf{0}$) capturada no início e obrigatoriamente recapturada ao final do bloco para verificação de deriva mecânica e fotométrica.
  - **$P_1 \dots P_6$ (Perturbações Isoladas Determinísticas)**: Translações e rotações puras no referencial $T_0$, amostradas em frações configuráveis dos limites $[l, u]$.
  - **$P_7$ (Perturbações Combinadas em 6-DoF via LHS)**: Geradas via `scipy.stats.qmc.LatinHypercube(d=6, optimization="random-cd", seed=seed)` ($N = 30$ amostras padrão) com fator de encolhimento de segurança `lhs_shrinkage_factor` (`0.75`).

- **Duas Configurações Pareadas no `Cam-App` (`auto` e `manual_calibrado`)**:
  - **`auto`**: Rotinas 3A do Android (AF, AE, AWB), pós-processamento ISP e estabilização óptica/eletrônica (OIS/EIS) ativados.
  - **`manual_calibrado`**: Exposição, sensibilidade, foco e balanço de branco fixos, com controles de pós-processamento e estabilização desativáveis configurados para OFF. Calibrado automaticamente a partir da leitura do `X-Camera-Metadata` em modo `auto` na pose inicial $P_0$.

---

## 📌 Estrutura do Projeto

```text
visual-servoing-dataset/
├── pyproject.toml              # Dependências Poetry (aether-rdk, opencv-contrib-python, scipy, numpy)
├── config.yaml                 # Configuração global (ChArUco, Abóbada, limites [l, u], P1..P6, LHS)
├── main.py                     # CLI do protocolo experimental (Fase 0 + Fases 1 a 3)
├── src/
│   └── visual_servoing_dataset/
│       ├── config.py           # Dataclasses de configuração
│       ├── schemas.py          # Esquemas de planos (SHA-256), calibração e metadados
│       ├── calibration/
│       │   ├── calibration_generator.py # Geração do ChArUco PNG/JSON e Abóbada Esférica (24 poses)
│       │   ├── calibration_solver.py    # CharucoDetector, calibrateCamera, calibrateHandEye, Rectifier, H_gt
│       │   └── calibration_manager.py   # Orquestração da Fase 0 e persistência de calibration_profile.json
│       ├── robot/
│       │   ├── denso_controller.py  # Wrapper da aether-rdk (controle cartesiano e por juntas)
│       │   ├── kinematics.py        # Conversões Pose <-> SE(3) 4x4, pós-multiplicação e inversa analítica
│       │   ├── pose_generator.py    # Gerador das famílias P0..P7 e Latin Hypercube Sampling (P7)
│       │   └── trajectories.py      # Adaptador de compatibilidade
│       ├── smartphone/
│       │   └── phone_controller.py  # Cliente HTTP Cam-App, extração X-Camera-Metadata e calibração P0
│       ├── collector/
│       │   ├── session_manager.py   # Fases 1 e 2 (explore-limits, generate-plan, verify-plan)
│       │   └── dataset_collector.py # Fase 3 (coleta pareada + retificação + H_gt + QA + CSV/JSON)
│       └── utils/
│           └── logger.py            # Logger formatado com Rich
└── tests/
    ├── test_protocol_se3_lhs.py     # Testes unitários das Fases 1-3 (SE(3), P0..P7, LHS)
    └── test_calibration_pipeline.py # Testes unitários e sintéticos da Fase 0 (Abóbada, K/D/Hand-Eye, H_gt)
```

---

## 🚀 Fluxo Operacional Completo (Fase 0 + Fases 1 a 3)

### Fase 0: Calibração Automatizada "2 em 1" (Intrínsecos + Hand-Eye via ChArUco)
1. **Gerar prancha ChArUco imprimível (PNG em alta resolução + `charuco_spec.json` para impressão 1:1)**:
   ```bash
   poetry run python main.py --generate-charuco
   ```
2. **Executar a rotina de calibração de 24 poses na Abóbada Esférica (estritamente em `manual_calibrado`)**:
   ```bash
   poetry run python main.py --calibrate-camera --verify-poses --interactive
   ```
   Gera `assets/calibration/calibration_profile.json` contendo $K$, $D$, $K_{\text{rect}}$, ${}^T T_C$ (Hand-Eye) e ${}^B T_{\text{board}}$.

### Fase 1: Exploração de Limites e Calibração em $P_0$ (`--explore-limits`)
```bash
poetry run python main.py --explore-limits --interactive
```

### Fase 2: Geração e Aprovação Supervisionada de Poses (`--generate-plan` e `--verify-plan`)
```bash
poetry run python main.py --generate-plan
poetry run python main.py --verify-plan --interactive
```

### Fase 3: Execução da Coleta Automática Pareada (`--collect`) e Retificação (`--rectify-dataset`)
```bash
# Coleta pareada (injeta calibration_profile.json, calcula B_T_C e H_gt e salva imagens retificadas)
poetry run python main.py --collect --plan assets/dataset/poses_plan.json --calibration-file assets/calibration/calibration_profile.json

# Retificar retroativamente um dataset existente
poetry run python main.py --rectify-dataset assets/dataset --calibration-file assets/calibration/calibration_profile.json
```

### Simulação Completa (`--dry-run`) e Testes Unitários (`pytest`)
```bash
# Testar Fase 0 + Fases 1 a 3 em modo de simulação
poetry run python main.py --generate-charuco --calibrate-camera --explore-limits --generate-plan --verify-plan --collect --dry-run

# Executar toda a suíte de testes unitários
poetry run pytest -v
```

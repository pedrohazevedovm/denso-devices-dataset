# Visual Servoing Dataset Collector (DENSO VS-050 + Android Cam-App)

Sistema em Python para automação da captura e organização de datasets de visão computacional e *Visual Servoing*. O sistema controla o robô industrial **DENSO VS-050** e sincroniza disparos fotográficos de smartphones Android utilizando exclusivamente o aplicativo dedicado **`Cam-App`** (servidor HTTP Camera2 HAL na porta 8080).

---

## 🌟 Principais Funcionalidades

- **Controle Nativo do Robô DENSO VS-050**: Comunicação via biblioteca `aether_rdk` (bCAP/CAO Provedor `CaoProv.DENSO.RC8`) utilizando objetos nativos de dados **`Pose`** (`X, Y, Z, Rx, Ry, Rz, Fig`) e **`Joint`** (`J1, J2, J3, J4, J5, J6`).
- **Integração Exclusiva com o `Cam-App` (Camera2 HAL HTTP)**:
  - Opera com apenas **duas configurações de câmera**:
    1. **`auto`**: Algoritmos 3A do Android (AF, AE, AWB), pós-processamento ISP e estabilizações (OIS/EIS) ativados.
    2. **`manual_calibrado`**: Controle manual determinístico estrito (3A, ISP, OIS e EIS desligados), com parâmetros de foco (`focus_distance`), ganho (`iso`) e tempo de integração (`exposure_time_ns`) calibrados automaticamente a partir da leitura do modo `auto` na pose inicial $P_0$ (ou fixados via `config.yaml`).
  - Extrai todos os **25+ metadados de hardware** do cabeçalho `X-Camera-Metadata` em cada foto (incluindo matriz intrínseca `camera_matrix_k`, `lens_intrinsic_calibration`, `sensor_exposure_time`, `sensor_sensitivity`, `lens_focus_distance`, `lens_state`, ganhos RGB e região de corte).
- **Gerador de Trajetórias e Variações $P_0 \dots P_4$**: A partir de cada pose base $P_0$, calcula variações controladas (`p0_reference.jpg`, `p1_trans_xy.jpg`, `p2_rot_z.jpg`, `p3_scale_z.jpg`, `p4_combined.jpg`).
- **Indexação Unificada de Metadados**: Gera `metadata.json` consolidando poses do robô, parâmetros calibrados e metadados completos de cada captura para `auto` e `manual_calibrado`.

---

## 🚀 Como Executar

### 1. Teste em Modo Simulação (Dry-Run)
```bash
poetry run python main.py --dry-run
```

### 2. Testar Conexão com o Robô DENSO VS-050
```bash
poetry run python main.py --test-robot
```

### 3. Testar Conexão e Calibração do `Cam-App`
Testa a comunicação HTTP (`GET /status`, `POST /config`, `GET /capture`), calibra o `manual_calibrado` a partir do `auto` e captura fotos de teste em ambas as configurações:
```bash
poetry run python main.py --test-phone
```

### 4. Executar Coleta Completa do Dataset
```bash
poetry run python main.py --collect --config config.yaml
```

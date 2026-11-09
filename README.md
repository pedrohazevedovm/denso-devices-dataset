# Visual Servoing Dataset Collector (DENSO VS-050 + Android Smartphone)

Sistema em Python de alta precisão projetado para automação da captura e organização de datasets de visão computacional e *Visual Servoing*. O sistema controla o robô industrial **DENSO VS-050** e sincroniza disparos fotográficos de **smartphones Android** acoplados como sensores visuais.

---

## 🌟 Principais Funcionalidades

- **Controle Nativo do Robô DENSO VS-050**: Comunicação via biblioteca `aether_rdk` (bCAP/CAO Provedor `CaoProv.DENSO.RC8`) utilizando objetos nativos de dados **`Pose`** (`X, Y, Z, Rx, Ry, Rz, Fig`) e **`Joint`** (`J1, J2, J3, J4, J5, J6`).
- **Controle Dual de Câmeras Android (IP Camera ou App Padrão)**:
  - **Modo `ip_camera` (Recomendado)**: Conexão direta via HTTP (NanoHTTPD na porta 8080) com aplicativo Android dedicado que bloqueia algoritmos 3A (Auto-Focus, Auto-Exposure, Auto-White Balance), estabilizações (OIS/EIS) e filtros ISP para máxima repetibilidade e zero latência. Suporta presets manuais (`focus_distance`, `iso`, `exposure_time_ns`) e extrai telemetria fotométrica direta do sensor.
  - **Modo `default_app` (Legado)**: Controle do aplicativo nativo de câmera do fabricante via ADB (`device_manager`).
- **Gerador de Trajetórias e Variações $P_0 \dots P_4$**: A partir de cada pose de enquadramento base $P_0$, calcula automaticamente 5 variações controladas:
  1. `p0_reference.jpg`: Pose de referência.
  2. `p1_trans_xy.jpg`: Translação apenas nos eixos $X$ e $Y$.
  3. `p2_rot_z.jpg`: Rotação apenas no eixo $R_z$ (Junta 6).
  4. `p3_scale_z.jpg`: Variação de profundidade/escala apenas no eixo $Z$.
  5. `p4_combined.jpg`: Variações combinadas de translação, profundidade e rotação.
- **Registro do Tool Center Point (TCP)**: Configuração do offset da ferramenta (`tcp_offset`) para acoplamento do suporte de smartphone na flange do robô.
- **Organização por Cenas e Câmeras**: Estruturação automática do dataset em subpastas legíveis (`scene_XX/cam_XX/`).
- **Indexação Unificada de Metadados**: Geração de `metadata.json` completo contendo poses comandadas e reais medidas, ângulos de juntas, telemetria fotométrica de câmera (ISO, exposição, foco), timestamps e caminhos das fotos.
- **Modo de Simulação (Dry-Run)**: Teste completo de fluxos, trajetórias e arquivos sem necessidade de hardware conectado.

---

## 📌 Estrutura do Projeto

```text
visual-servoing-dataset/
├── pyproject.toml              # Dependências Poetry (aether-rdk, device-manager, pyyaml, requests, etc.)
├── config.yaml                 # Arquivo de configuração principal (IP, modo de câmera, TCP, variações, cenas)
├── main.py                     # CLI principal (coleta, testes isolados, dry-run)
├── src/
│   └── visual_servoing_dataset/
│       ├── config.py           # Dataclasses de configuração (suporte a dual camera mode e objetos Pose/Joint)
│       ├── robot/
│       │   ├── denso_controller.py  # Wrapper da aether-rdk (controle cartesiano e por juntas)
│       │   └── trajectories.py      # Gerador das variações P0..P4 em objetos Pose
│       ├── smartphone/
│       │   └── phone_controller.py  # Controlador unificado de câmeras (IP Camera HTTP + App Padrão ADB)
│       ├── collector/
│       │   └── dataset_collector.py # Orquestrador do dataset e gerador do metadata.json
│       └── utils/
│           └── logger.py            # Logger formatado com biblioteca Rich
└── assets/
    └── dataset/                 # Saída estruturada do dataset gerado
        ├── metadata.json
        ├── scene_01/
        │   ├── cam_01_moto_edge_40/
        │   │   ├── p0_reference.jpg
        │   │   └── ...
        └── scene_02/
            └── ...
```

---

## 🛠️ Pré-requisitos e Instalação

1. **Requisitos de Software**:
   - Python `>=3.10, <3.14`
   - [Poetry](https://python-poetry.org/)

2. **Instalação do Projeto**:
   ```bash
   poetry install
   ```

3. **Pré-requisitos de Hardware**:
   - **Robô DENSO VS-050**: Controlador DENSO RC8 com servidor bCAP habilitado e endereço IP configurado (ex: `192.168.160.227`).
   - **Smartphones Android**:
     - *No modo `ip_camera`*: Aplicativo `Headless Android IP Camera` instalado e aberto na tela (porta 8080) na mesma rede Wi-Fi/LAN.
     - *No modo `default_app`*: Depuração USB habilitada nas opções de desenvolvedor e conectado via cabo USB.

---

## ⚙️ Arquivo de Configuração (`config.yaml`)

O arquivo `config.yaml` define todos os parâmetros operacionais:

```yaml
dry_run: true  # Defina como false para conectar ao hardware real

robot:
  ip_address: "192.168.160.227"
  workspace_name: "VS_WORKSPACE"
  control_name: "DENSO_VS050"
  speed: 20.0                  # Velocidade do braço (0 a 100%)
  accel: 20.0                  # Aceleração (0 a 100%)
  decel: 20.0                  # Desaceleração (0 a 100%)
  stabilization_delay_sec: 1.5 # Pausa de amortecimento antes da foto
  
  tcp_offset:                  # Deslocamento do Tool Center Point (TCP)
    x: 0.0                     # mm
    y: 0.0                     # mm
    z: 0.0                     # mm
    rx: 0.0                    # graus
    ry: 0.0                    # graus
    rz: 0.0                    # graus
    tag: "SMARTPHONE_TOOL"

smartphone:
  mode: "ip_camera"            # "ip_camera" (novo app Camera2) ou "default_app" (app nativo ADB)

  # Configurações do modo "ip_camera" (HTTP / porta 8080)
  ips:
    - "192.168.158.48"
  port: 8080
  default_preset: "1_home_visao_geral"
  stabilization_delay_sec: 0.5
  presets:
    "1_home_visao_geral":
      focus_distance: 1.66     # Dioptrias
      iso: 100
      exposure_time_ns: 8333333 # ~8.33ms
    "2_aproximacao_fina":
      focus_distance: 5.0
      iso: 100
      exposure_time_ns: 10000000

  # Configurações do modo "default_app" (legado ADB)
  serials:
    - "auto"
  capture_delay_sec: 2.0

  # Mapeamento comum de identificadores para nomes de pasta
  camera_names:
    "192.168.158.48": "cam_01_moto_edge_40"
    "sim_serial_01": "cam_01_moto_edge_40"

variations:
  delta_trans_xy:
    dx: 20.0
    dy: 15.0
  delta_rot_z: 15.0
  delta_scale_z: -30.0
  delta_combined:
    dx: 15.0
    dy: 10.0
    dz: -20.0
    drz: 10.0

scenes:
  - name: "scene_01"
    description: "Enquadramento do Cenário 01 - Visão Geral"
    camera_preset: "1_home_visao_geral"
    p0_pose:
      cartesian: [440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1]

  - name: "scene_02"
    description: "Enquadramento do Cenário 02 - Aproximação Fina"
    camera_preset: "2_aproximacao_fina"
    p0_pose:
      cartesian: [440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1]

dataset:
  output_dir: "assets/dataset"
  save_raw_photos: true
```

---

## 🚀 Como Executar

### 1. Teste em Modo Simulação (Dry-Run)
Valida a geração de trajetórias, objetos `Pose`/`Joint` e árvore de arquivos de saída sem depender de hardware físico:
```bash
poetry run python main.py --dry-run
```

### 2. Testar Conexão com o Robô DENSO VS-050
Verifica a conexão bCAP e a leitura da pose atual do robô:
```bash
poetry run python main.py --test-robot
```

### 3. Testar Câmeras dos Smartphones Android
Verifica a captura de foto em todos os smartphones conectados via ADB:
```bash
poetry run python main.py --test-phone
```

### 4. Executar Coleta Completa do Dataset
Inicia a rotina de captura de todas as cenas e variações $P_0 \dots P_4$:
```bash
poetry run python main.py --collect --config config.yaml
```

---

## 📄 Formato do Arquivo de Metadados (`metadata.json`)

O arquivo `assets/dataset/metadata.json` armazena as informações consolidadas da sessão:

```json
{
  "dataset_version": "1.0",
  "generated_at": "2026-08-25T19:38:10.315549",
  "dry_run": false,
  "robot_info": {
    "model": "DENSO VS-050",
    "ip_address": "192.168.0.1",
    "tcp_offset": { "x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0, "tag": "SMARTPHONE_TOOL" }
  },
  "cameras": {
    "0123456789": "cam_01_moto_edge_40",
    "9876543210": "cam_02_moto_g13"
  },
  "scenes": {
    "scene_01": {
      "name": "scene_01",
      "p0_base_pose": [440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1],
      "waypoints": [
        {
          "waypoint_id": "p0_reference",
          "filename": "p0_reference.jpg",
          "variation_type": "reference",
          "commanded_cartesian_pose": [440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1],
          "actual_cartesian_pose": [440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1],
          "actual_joint_pose": [12.5, -45.0, 80.2, 0.0, 54.8, 150.0],
          "delta_applied": { "dx": 0.0, "dy": 0.0, "dz": 0.0, "drz": 0.0 },
          "timestamp": "2026-08-25T19:38:02.980358",
          "captures": {
            "cam_01_moto_edge_40": "scene_01/cam_01_moto_edge_40/p0_reference.jpg",
            "cam_02_moto_g13": "scene_01/cam_02_moto_g13/p0_reference.jpg"
          }
        }
      ]
    }
  }
}
```

# Visual Servoing Dataset Collector (DENSO VS-050 + Android Smartphone)

Sistema em Python de alta precisão projetado para automação da captura e organização de datasets de visão computacional e *Visual Servoing*. O sistema controla o robô industrial **DENSO VS-050** e sincroniza disparos fotográficos de **smartphones Android** acoplados como sensores visuais.

---

## 🌟 Principais Funcionalidades

- **Controle Nativo do Robô DENSO VS-050**: Comunicação via biblioteca `aether_rdk` (bCAP/CAO Provedor `CaoProv.DENSO.RC8`) utilizando objetos nativos de dados **`Pose`** (`X, Y, Z, Rx, Ry, Rz, Fig`) e **`Joint`** (`J1, J2, J3, J4, J5, J6`).
- **Controle Automático de Smartphones Android**: Integração via `device_manager` (ADB) para descoberta de dispositivos USB/IP, controle do app de câmera, disparo e download automático de imagens.
- **Gerador de Trajetórias e Variações $P_0 \dots P_4$**: A partir de cada pose de enquadramento base $P_0$, calcula automaticamente 5 variações controladas:
  1. `p0_reference.jpg`: Pose de referência.
  2. `p1_trans_xy.jpg`: Translação apenas nos eixos $X$ e $Y$.
  3. `p2_rot_z.jpg`: Rotação apenas no eixo $R_z$ (Junta 6).
  4. `p3_scale_z.jpg`: Variação de profundidade/escala apenas no eixo $Z$.
  5. `p4_combined.jpg`: Variações combinadas de translação, profundidade e rotação.
- **Registro do Tool Center Point (TCP)**: Configuração do offset da ferramenta (`tcp_offset`) para acoplamento do suporte de smartphone na flange do robô.
- **Organização por Cenas e Câmeras**: Estruturação automática do dataset em subpastas legíveis (`scene_XX/cam_XX/`).
- **Indexação Unificada de Metadados**: Geração de `metadata.json` completo contendo poses comandadas e reais medidas, ângulos de juntas, timestamps e caminhos das fotos.
- **Modo de Simulação (Dry-Run)**: Teste completo de fluxos, trajetórias e arquivos sem necessidade de hardware conectado.

---

## 📌 Estrutura do Projeto

```text
visual-servoing-dataset/
├── pyproject.toml              # Dependências Poetry (aether-rdk, device-manager, pyyaml, rich, etc.)
├── config.yaml                 # Arquivo de configuração principal (IP, TCP, variações, cenas)
├── main.py                     # CLI principal (coleta, testes isolados, dry-run)
├── src/
│   └── visual_servoing_dataset/
│       ├── config.py           # Dataclasses de configuração e suporte aos objetos Pose/Joint
│       ├── robot/
│       │   ├── denso_controller.py  # Wrapper da aether-rdk (controle cartesiano e por juntas)
│       │   └── trajectories.py      # Gerador das variações P0..P4 em objetos Pose
│       ├── smartphone/
│       │   └── phone_controller.py  # Wrapper da device-manager (controle ADB e download de fotos)
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
        │   │   ├── p1_trans_xy.jpg
        │   │   ├── p2_rot_z.jpg
        │   │   ├── p3_scale_z.jpg
        │   │   └── p4_combined.jpg
        │   └── cam_02_moto_g13/
        │       └── ...
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
   - **Robô DENSO VS-050**: Controlador DENSO RC8 com servidor bCAP habilitado e endereço IP configurado (ex: `192.168.0.1`).
   - **Smartphones Android**: **Depuração USB** habilitada nas opções de desenvolvedor e conectados via cabo USB ao computador.

---

## ⚙️ Arquivo de Configuração (`config.yaml`)

O arquivo `config.yaml` define todos os parâmetros operacionais:

```yaml
dry_run: true  # Defina como false para conectar ao hardware real

robot:
  ip_address: "192.168.0.1"
  workspace_name: "VS_WORKSPACE"
  control_name: "DENSO_VS050"
  speed: 20.0                  # Velocidade do braço (0 a 100%)
  accel: 20.0                  # Aceleração (0 a 100%)
  decel: 20.0                  # Desaceleração (0 a 100%)
  stabilization_delay_sec: 1.5 # Pausa de estabilização antes da foto (em segundos)
  
  tcp_offset:                  # Deslocamento do Tool Center Point (TCP)
    x: 0.0                     # mm
    y: 0.0                     # mm
    z: 0.0                     # mm
    rx: 0.0                    # graus
    ry: 0.0                    # graus
    rz: 0.0                    # graus
    tag: "SMARTPHONE_TOOL"

smartphone:
  serials:
    - "auto"                   # "auto" para detectar todos os celulares USB via ADB
  camera_names:                # Nomes amigáveis para as pastas dos dispositivos
    "sim_serial_01": "cam_01_moto_edge_40"
    "sim_serial_02": "cam_02_moto_g13"
  capture_delay_sec: 2.0

variations:
  delta_trans_xy:              # P1 (Translação XY)
    dx: 20.0                   # mm
    dy: 15.0                   # mm
  delta_rot_z: 15.0            # P2 (Rotação Rz / Junta 6 em graus)
  delta_scale_z: -30.0         # P3 (Profundidade Z em mm)
  delta_combined:              # P4 (Combinada)
    dx: 15.0                   # mm
    dy: 10.0                   # mm
    dz: -20.0                  # mm
    drz: 10.0                  # graus

scenes:
  - name: "scene_01"
    description: "Enquadramento do Cenário 01"
    p0_pose:
      cartesian: [440.0, 60.0, 585.0, 16.0, -22.0, 150.0, 1]  # [X, Y, Z, Rx, Ry, Rz, Fig]

  - name: "scene_02"
    description: "Enquadramento do Cenário 02"
    p0_pose:
      cartesian: [450.0, 70.0, 590.0, 16.0, -22.0, 150.0, 1]

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

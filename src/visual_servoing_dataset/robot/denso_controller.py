import time
from typing import List, Optional, Union, Dict, Any
from ..config import RobotConfig, TCPOffset, Pose, Joint, ensure_pose, ensure_joint
from ..utils.logger import logger

try:
    from aether_rdk import DensoRobotAPI
    from aether_rdk.datatypes.offset_3d import Offset3D
    AETHER_RDK_AVAILABLE = True
except ImportError:
    AETHER_RDK_AVAILABLE = False
    Offset3D = None


class DensoRobotController:
    """Controlador para o robô industrial DENSO VS-050 usando objetos Pose e Joint da aether-rdk (bCAP/CAO)."""

    def __init__(self, config: RobotConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run or not AETHER_RDK_AVAILABLE
        self.api: Optional[Any] = None
        self.is_connected: bool = False
        self._current_sim_pose: Pose = ensure_pose([300.0, 0.0, 400.0, 180.0, 0.0, 0.0, 0])
        self._current_sim_joints: Joint = ensure_joint([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def connect(self) -> bool:
        """Conecta ao controlador DENSO VS-050, habilita motores e configura o Tool (TCP)."""
        if self.dry_run:
            logger.info("[DRY-RUN] Simulação de conexão com robô DENSO VS-050 bem-sucedida.")
            self.is_connected = True
            return True

        if not AETHER_RDK_AVAILABLE:
            logger.error("A biblioteca 'aether_rdk' não está disponível no ambiente!")
            return False

        options = f"Server={self.config.ip_address}"
        logger.info(f"Conectando ao robô DENSO VS-050 em {self.config.ip_address}...")

        try:
            self.api = DensoRobotAPI(
                workspace_name=self.config.workspace_name,
                control_name=self.config.control_name,
                options=options
            )
            connected = self.api.connect()
            if connected:
                self.is_connected = True
                logger.info("Conexão com DENSO VS-050 estabelecida.")

                # Habilitar motor
                if not self.api.motor_enabled():
                    self.api.motor_on()
                    logger.info("Motores do robô habilitados (Motor ON).")

                # Ajustar velocidade
                self.api.set_arm_speed(
                    self.config.speed,
                    self.config.accel,
                    self.config.decel
                )
                logger.info(f"Velocidade configurada: {self.config.speed}%")

                # Configurar TCP (Tool Offset)
                self.setup_tcp(self.config.tcp_offset)
                return True
            else:
                logger.error("Falha ao estabelecer conexão bCAP com controlador DENSO.")
                return False

        except Exception as e:
            logger.error(f"Exceção ao conectar com DENSO VS-050: {e}")
            self.is_connected = False
            return False

    def setup_tcp(self, tcp: TCPOffset) -> bool:
        """Configura a referência da ferramenta (Tool Frame / TCP) no robô DENSO."""
        if self.dry_run or not self.api:
            logger.info(f"[DRY-RUN] TCP Offset configurado: {tcp.to_dict()}")
            return True

        try:
            offset = Offset3D(tcp.x, tcp.y, tcp.z, tcp.rx, tcp.ry, tcp.rz)
            self.api.create_tool_reference(offset, tcp.tag)
            self.api.set_current_tool_by_tag(tcp.tag)
            logger.info(f"Ferramenta TCP '{tcp.tag}' definida no robô: X={tcp.x}, Y={tcp.y}, Z={tcp.z}")
            return True
        except Exception as e:
            logger.warning(f"Não foi possível registrar TCP customizado '{tcp.tag}': {e}")
            return False

    def move_cartesian(self, pose: Union[Pose, List[float], Dict[str, float]]) -> bool:
        """Move o braço para a pose cartesiana desejada representada pelo objeto Pose."""
        pose_obj: Pose = ensure_pose(pose)
        logger.info(
            f"Movendo robô para objeto Pose: X={pose_obj.x:.2f}, Y={pose_obj.y:.2f}, Z={pose_obj.z:.2f}, "
            f"Rx={pose_obj.rx:.2f}, Ry={pose_obj.ry:.2f}, Rz={pose_obj.rz:.2f}, Fig={pose_obj.fig}"
        )

        if self.dry_run:
            self._current_sim_pose = pose_obj
            time.sleep(0.2)
            return True

        if not self.is_connected or not self.api:
            logger.error("Erro: Robô DENSO não está conectado.")
            return False

        try:
            success = self.api.move_cartesian(pose_obj)
            if self.config.stabilization_delay_sec > 0:
                time.sleep(self.config.stabilization_delay_sec)
            return success
        except Exception as e:
            logger.error(f"Erro na movimentação cartesiana com objeto Pose: {e}")
            return False

    def move_joints(self, joints: Union[Joint, List[float], Dict[str, float]]) -> bool:
        """Move o robô para ângulos de juntas representados pelo objeto Joint."""
        joint_obj: Joint = ensure_joint(joints)
        logger.info(
            f"Movendo robô para objeto Joint: J1={joint_obj.joint_1:.2f}, J2={joint_obj.joint_2:.2f}, "
            f"J3={joint_obj.joint_3:.2f}, J4={joint_obj.joint_4:.2f}, J5={joint_obj.joint_5:.2f}, J6={joint_obj.joint_6:.2f}"
        )

        if self.dry_run:
            self._current_sim_joints = joint_obj
            time.sleep(0.2)
            return True

        if not self.is_connected or not self.api:
            logger.error("Erro: Robô DENSO não está conectado.")
            return False

        try:
            success = self.api.move_joints(joint_obj)
            if self.config.stabilization_delay_sec > 0:
                time.sleep(self.config.stabilization_delay_sec)
            return success
        except Exception as e:
            logger.error(f"Erro na movimentação por juntas com objeto Joint: {e}")
            return False

    def get_cartesian_pose(self) -> Pose:
        """Retorna a pose cartesiana atual como objeto Pose."""
        if self.dry_run or not self.api:
            return self._current_sim_pose

        try:
            pose_obj = self.api.get_cartesian_pose()
            if pose_obj:
                return pose_obj
        except Exception as e:
            logger.warning(f"Erro ao obter objeto Pose real: {e}")

        return self._current_sim_pose

    def get_joints_pose(self) -> Joint:
        """Retorna a pose de juntas atual como objeto Joint."""
        if self.dry_run or not self.api:
            return self._current_sim_joints

        try:
            joint_obj = self.api.get_joints_pose()
            if joint_obj:
                return joint_obj
        except Exception as e:
            logger.warning(f"Erro ao obter objeto Joint real: {e}")

        return self._current_sim_joints

    def errors_cleanup(self) -> bool:
        """Limpa falhas/erros do controlador DENSO."""
        if self.dry_run or not self.api:
            logger.info("[DRY-RUN] Limpeza de falhas executada.")
            return True
        try:
            return self.api.errors_cleanup()
        except Exception as e:
            logger.error(f"Erro ao tentar limpar erros do controlador: {e}")
            return False

    def emergency_stop(self) -> bool:
        """Comando de parada de emergência."""
        logger.warning("PARADA DE EMERGÊNCIA SOLICITADA!")
        if self.dry_run or not self.api:
            return True
        try:
            return self.api.emergency_stop()
        except Exception as e:
            logger.error(f"Erro na parada de emergência: {e}")
            return False

    def disconnect(self) -> bool:
        """Desconecta do robô e remove o workspace bCAP."""
        logger.info("Desconectando do robô DENSO VS-050...")
        if self.dry_run:
            self.is_connected = False
            return True

        if self.api:
            try:
                self.api.motor_off()
                self.api.disconnect()
                self.api.remove_workspace()
            except Exception as e:
                logger.warning(f"Erro ao desconectar: {e}")

        self.is_connected = False
        logger.info("Desconectado com sucesso.")
        return True
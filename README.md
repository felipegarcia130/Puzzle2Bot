# 🤖 MCL + A\* Autonomous Navigation — PuzzleBot

> Monte Carlo Localization con backup EKF y planificación A\* para un robot diferencial sobre ROS 2.

---

## Descripción

Sistema de navegación autónoma para el **PuzzleBot** (robot diferencial) que combina:

- **MCL (Monte Carlo Localization)** con filtro de partículas adaptativo (KLD-sampling) para estimación de pose en un mapa pre-construido.
- **EKF de respaldo** que mantiene la estimación de posición cuando el LiDAR pierde confianza (obstáculos no mapeados, zonas sin features).
- **Máquina de estados MCL ↔ EKF** con transiciones suaves y modo de sincronización.
- **Módulos adaptativos**: modelo de movimiento RAM (Robust Adaptive Motion Model) y selector de configuración del sensor model via Bandit (UCB).
- **Planificador A\*** sobre mapa inflado con evasión reactiva de obstáculos usando LiDAR.

---

## Arquitectura

```
/odom  ──────────────────────────────────────────────────────┐
                                                              ▼
                                                    ┌─────────────────┐
                                                    │  odom_callback  │
                                                    │  RAM motion     │
                                                    │  EKF predict    │
                                                    └────────┬────────┘
                                                             │ partículas propagadas
/scan  ──────────────────────────────────────────────────────┤
                                                             ▼
                                                   ┌──────────────────┐
                                                   │  scan_callback   │
                                                   │  Ray casting     │
                                                   │  Beam model      │
                                                   │  Bandit selector │
                                                   │  FSM MCL↔EKF     │
                                                   └────────┬─────────┘
                                                            │
                                               /mcl_pose ───┘
                                                            │
/goal_pose ──────────────────────────────────────────────── ▼
                                                  ┌──────────────────┐
                                                  │   astar_node     │
                                                  │   A* + inflado   │
                                                  │   Control P      │
                                                  │   Evasión LiDAR  │
                                                  └────────┬─────────┘
                                                           │
                                               /cmd_vel ───┘
```

---

## Nodos

### `mcl_node.py`

Estimación de pose del robot mediante filtro de partículas con respaldo EKF.

| Suscribe | Tipo | Descripción |
|---|---|---|
| `/odom` | `nav_msgs/Odometry` | Odometría para propagar partículas y predecir EKF |
| `/scan` | `sensor_msgs/LaserScan` | Mediciones LiDAR para actualizar pesos |

| Publica | Tipo | Descripción |
|---|---|---|
| `/mcl_pose` | `geometry_msgs/PoseStamped` | Pose estimada (MCL, EKF o fusionada) |
| `/map` | `nav_msgs/OccupancyGrid` | Mapa cargado desde `.npy` |

**Máquina de estados:**

| Estado | Condición de entrada | Fuente de pose |
|---|---|---|
| `MCL` | Convergencia > `CONV_HIGH` | Mediana de partículas |
| `EKF` | Convergencia < `CONV_LOW` por ≥ N scans | Predicción EKF (odometría) |
| `SYNC` | MCL recuperándose | Fusión ponderada MCL + EKF |

---

### `astar_node.py`

Planificación de trayectoria y control de movimiento.

| Suscribe | Tipo | Descripción |
|---|---|---|
| `/mcl_pose` | `geometry_msgs/PoseStamped` | Pose actual del robot |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Goal publicado desde RViz |
| `/scan` | `sensor_msgs/LaserScan` | Evasión reactiva de obstáculos |

| Publica | Tipo | Descripción |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Comandos de velocidad |
| `/planned_path` | `nav_msgs/Path` | Ruta completa para RViz |
| `/current_waypoint` | `geometry_msgs/PoseStamped` | Waypoint activo |

---

## Parámetros clave

### Mapa

| Parámetro | Valor | Descripción |
|---|---|---|
| `MAP_X_MIN/MAX` | -4.0 / 5.0 m | Extensión del mapa en X |
| `MAP_Y_MIN/MAX` | -3.0 / 6.0 m | Extensión del mapa en Y |
| `RESOLUTION` | 0.05 m/px | Resolución del mapa |

### MCL

| Parámetro | Valor | Descripción |
|---|---|---|
| `N` | 200 | Número máximo de partículas |
| `N_MIN` | 30 | Mínimo con KLD-sampling |
| `P_NOISE` | 0.8 | Peso del modelo gaussiano en beam model |
| `CONV_LOW` | 0.25 | Umbral bajo de convergencia → failover a EKF |
| `CONV_HIGH` | 0.55 | Umbral alto → MCL recupera control |
| `SCANS_TO_FAILOVER` | 5 | Scans consecutivos bajo umbral para ceder a EKF |

### A\*

| Parámetro | Valor | Descripción |
|---|---|---|
| `INFLATE_RADIUS` | 6 px (0.3 m) | Radio de inflado de obstáculos |
| `WAYPOINT_TOL` | 0.25 m | Tolerancia para avanzar al siguiente waypoint |
| `GOAL_TOL` | 0.35 m | Tolerancia para declarar goal alcanzado |
| `LIN_VEL` | 0.07 m/s | Velocidad lineal nominal |
| `ANG_VEL` | 0.07 rad/s | Velocidad angular máxima |

---

## Requisitos

- ROS 2 Humble / Jazzy
- Python ≥ 3.10
- `numpy`, `opencv-python`, `scipy`
- RPLIDAR A1 (o simulación en Gazebo)
- Mapa pre-construido en formato `.npy` (salida del nodo SLAM)

---

## Uso

### 1. Colocar el mapa

```bash
# El mapa debe estar en:
ros2_ws/src/mcl_robot/maps/slam_map.npy
```

### 2. Lanzar los nodos

#### 🖥️ Simulación (Gazebo)

```bash
# Terminal 1 — Simulación PuzzleBot
ros2 launch puzzlebot_gazebo puzzle_sim_pb.launch.py

# Terminal 2 — Localización MCL + EKF
ros2 run mcl_robot mcl_node

# Terminal 3 — Planificador A* + control
ros2 run mcl_robot astar_node
```

#### 🤖 Robot real

```bash
# Terminal 1 — Odometría
ros2 run mcl_robot odom_node

# Terminal 2 — Localización MCL + EKF
ros2 run mcl_robot mcl_node

# Terminal 3 — Planificador A* + control
ros2 run mcl_robot astar_node
```

### 3. Visualizar en RViz

Agregar los topics:
- `/map` → OccupancyGrid
- `/mcl_pose` → PoseStamped
- `/planned_path` → Path
- `/current_waypoint` → PoseStamped

Usar **2D Goal Pose** para enviar un goal.

---

## Módulos internos

### EKFBackup

EKF de 3 estados `[x, y, θ]` con modelo cinemático diferencial no-lineal. Jacobiano analítico. Ruido de proceso proporcional al movimiento (`Q_trans`, `Q_rot`). Ruido de observación adaptativo según convergencia del MCL.

### RobustAdaptiveMotionModel (RAM)

Adapta la covarianza del ruido de movimiento en línea usando un esquema de Robbins-Monro. Objetivo de tasa de aceptación: 35%.

### BanditSensorSelector

Selecciona la configuración óptima del beam model (σ_hit, skip de rayos) usando UCB con varianza. Permite ajustar automáticamente el balance velocidad/precisión del sensor model.

---

## Notas

- En entornos con obstáculos dinámicos o zonas sin features, el EKF mantiene la trayectoria relativa mientras el MCL se recupera.
- Las partículas se redistribuyen alrededor de la estimación EKF durante el estado `EKF`, acelerando la reconvergencia.
- El ray casting vectorizado opera en espacio de píxeles para máximo rendimiento en NumPy.

---

## Autor

**Felipe Garcia** — Robotics & Digital Systems Engineering, Tecnológico de Monterrey  
Proyecto desarrollado como parte del curso *TE3003B — Integración de Robótica*.

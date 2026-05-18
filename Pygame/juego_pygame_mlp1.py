import os
import csv
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Opcional: para graficar los datos en 2D y 3D
import matplotlib
# Configuramos backend para ventanas interactivas
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass  # Usa el backend por defecto
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.ion()

BASE_W, BASE_H = 1080, 720
WINDOW_FRACTION = 0.97
EXTRA_SCALE = 1.1

@dataclass
class Sample:
    velocidad_bala: float
    distancia: float
    altura_bala: float
    accion: int  # 0: nada, 1: saltar, 2: agacharse

class Juego:
    def __init__(self) -> None:
        pygame.init()

        self._flags = 0
        self._fullscreen = False

        start_w = BASE_W
        start_h = BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Bala + salto + agacharse + MLP + Tree")

        self.BLANCO = (255, 255, 255)
        self.NEGRO = (0, 0, 0)
        self.GRIS = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)

        self.corriendo = True
        self.modo_auto = False
        self.usar_tree = False # False = MLP, True = Tree

        self.datos_modelo: List[Sample] = []
        self.modelo_mlp: Optional[MLPClassifier] = None
        self.modelo_tree: Optional[DecisionTreeClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.modelo_entrenado = False
        self.clase_unica: Optional[int] = None
        self.ultima_accion_auto: Optional[int] = None

        self.decision_window = 500
        self._decision_frame_counter = 0

        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self.margin = 50
        self.ground_y = self.h - 100
        self.player_size = (32, 48)
        self.bullet_size = (16, 16)
        self.ship_size = (64, 64)
        self.fondo_speed = 3

        self.salto = False
        self.en_suelo = True
        self.agachado = False
        self.salto_vel_inicial = 15.0
        self.gravedad = 1.0
        self.salto_vel = self.salto_vel_inicial

        self.current_frame = 0
        self.frame_speed = 10
        self.frame_count = 0

        self.velocidad_bala = -12
        self.bala_disparada = False
        self.fondo_x1 = 0
        self.fondo_x2 = start_w

        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)
        self.scale = min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE
        self.scale = max(1.0, self.scale)

        self.margin = int(50 * self.scale)
        ground_offset = int(100 * self.scale)
        self.ground_y = self.h - ground_offset

        self.player_size = (int(32 * self.scale), int(48 * self.scale))
        self.bullet_size = (int(16 * self.scale), int(16 * self.scale))
        self.ship_size = (int(64 * self.scale), int(64 * self.scale))
        self.fondo_speed = max(1, int(2 * self.scale))

        self.salto_vel_inicial = 15 * self.scale
        self.gravedad = 1 * self.scale
        self.salto_vel = self.salto_vel_inicial

        self.decision_window = int(500 * self.scale)

        self.fuente = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        if reset_positions or not hasattr(self, "jugador"):
            self.jugador = pygame.Rect(self.margin, self.ground_y, self.player_size[0], self.player_size[1])
            self.bala = pygame.Rect(
                self.w - self.margin,
                self.ground_y + int(10 * self.scale),
                self.bullet_size[0],
                self.bullet_size[1],
            )
            self.nave = pygame.Rect(
                self.w - int(100 * self.scale),
                self.ground_y,
                self.ship_size[0],
                self.ship_size[1],
            )

    def _cargar_assets(self) -> None:
        def safe_load(path: str, size: Tuple[int, int], fallback_color=(200, 200, 200, 255)) -> pygame.Surface:
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size)
            except Exception:
                surf = pygame.Surface(size, pygame.SRCALPHA)
                surf.fill(fallback_color)
                return surf

        base = os.path.dirname(__file__)
        self.jugador_frames = [
            safe_load(os.path.join(base, "assets/sprites/mono_frame_1.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_2.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_3.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_4.png"), self.player_size),
        ]
        self.bala_img = safe_load(
            os.path.join(base, "assets/sprites/purple_ball.png"),
            self.bullet_size,
            (160, 120, 255, 255),
        )
        self.fondo_img = safe_load(
            os.path.join(base, "assets/game/fondo2.png"),
            (self.w, self.h),
            (40, 40, 40, 255),
        )
        self.nave_img = safe_load(
            os.path.join(base, "assets/game/ufo.png"),
            self.ship_size,
            (140, 255, 200, 255),
        )

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            w = info.current_w or self.w
            h = info.current_h or self.h
            self.pantalla = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
            self._apply_resolution(w, h, reset_positions=True)
        else:
            self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
            self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    def _reset_estado_juego(self) -> None:
        self.jugador.x, self.jugador.y = self.margin, self.ground_y
        self.jugador.height = self.player_size[1]
        self.nave.x, self.nave.y = self.w - int(100 * self.scale), self.ground_y
        self.bala.x = self.w - self.margin
        self.bala.y = self.ground_y + int(10 * self.scale)
        self.bala_disparada = False
        self.velocidad_bala = int(-10 * self.scale)
        self.salto = False
        self.en_suelo = True
        self.agachado = False
        self.salto_vel = self.salto_vel_inicial
        self._decision_frame_counter = 0
        self.fondo_x1 = 0
        self.fondo_x2 = self.w

    def _reset_modelo(self) -> None:
        self.modelo_mlp = None
        self.modelo_tree = None
        self.scaler = None
        self.modelo_entrenado = False
        self.clase_unica = None

    def exportar_datos_csv(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para exportar."
        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "datos_mlp.csv")
        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["velocidad_bala", "distancia", "altura_bala", "accion"])
                for s in self.datos_modelo:
                    writer.writerow([s.velocidad_bala, s.distancia, s.altura_bala, s.accion])
        except Exception as e:
            return f"Error al guardar CSV: {e}"
        return f"CSV guardado en datos_mlp.csv ({len(self.datos_modelo)} filas)."

    def graficar_datos_2d(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para graficar."
        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        # Colores: 0=azul (nada), 1=rojo (salto), 2=verde (agacharse)
        colores = {0: "blue", 1: "red", 2: "green"}
        cs = [colores[s.accion] for s in self.datos_modelo]

        fig_num = plt.figure("Datos Modelo - 2D", figsize=(8, 6)).number
        plt.figure(fig_num)
        plt.clf()
        ax = plt.gca()
        ax.scatter(xs, ys, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia jugador-bala")
        ax.set_ylabel("Velocidad bala")
        ax.set_title("Datos (Azul=Nada, Rojo=Salto, Verde=Agacharse)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()
        return "Mostrando gráfica 2D interactiva."

    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            self.velocidad_bala = int(random.randint(-12, -6) * self.scale)
            # Altura aleatoria para obligar a agacharse o saltar
            # Alta: requiere agacharse. Baja: requiere saltar.
            altura_baja = self.ground_y + int(25 * self.scale)
            altura_alta = self.ground_y + int(5 * self.scale)
            
            self.bala.y = random.choice([altura_baja, altura_alta])
            self.bala_disparada = True

    def reset_bala(self) -> None:
        self.bala.x = self.w - self.margin
        self.bala_disparada = False

    def iniciar_salto(self) -> None:
        if self.en_suelo and not self.agachado:
            self.salto = True
            self.en_suelo = False

    def manejar_salto(self) -> None:
        if self.salto:
            self.jugador.y -= int(self.salto_vel)
            self.salto_vel -= self.gravedad
            if self.jugador.y >= self.ground_y:
                self.jugador.y = self.ground_y
                self.salto = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo = True

    def iniciar_agacharse(self) -> None:
        if self.en_suelo and not self.agachado:
            self.agachado = True
            # Reducimos la altura a la mitad y bajamos el rectángulo
            self.jugador.height = self.player_size[1] // 2
            self.jugador.y = self.ground_y + self.player_size[1] // 2

    def levantar(self) -> None:
        if self.agachado:
            self.agachado = False
            self.jugador.height = self.player_size[1]
            self.jugador.y = self.ground_y

    def registrar_decision_manual(self) -> None:
        if not self.bala_disparada:
            return
        distancia = abs(self.jugador.x - self.bala.x)
        
        if not self.en_suelo:
            accion = 1 # Salto
        elif self.agachado:
            accion = 2 # Agacharse
        else:
            accion = 0 # Nada

        self.datos_modelo.append(
            Sample(
                velocidad_bala=float(self.velocidad_bala),
                distancia=float(distancia),
                altura_bala=float(self.bala.y),
                accion=accion,
            )
        )

    def entrenar_modelo(self) -> Tuple[bool, str]:
        samples = list(self.datos_modelo)
        if len(samples) < 80:
            return False, "Necesitas más datos (>= 80). Juega en MANUAL."
        X = [[s.velocidad_bala, s.distancia, s.altura_bala] for s in samples]
        y = [s.accion for s in samples]
        clases = sorted(set(y))
        if len(clases) < 2:
            self._reset_modelo()
            self.clase_unica = int(clases[0])
            self.modelo_entrenado = True
            return True, f"Modelo trivial entrenado: Siempre hace la acción {self.clase_unica}."
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Entrenar MLP
        clf_mlp = MLPClassifier(
            hidden_layer_sizes=(5, 5),
            activation="relu",
            solver="adam",
            max_iter=50000,
            random_state=42,
        )
        clf_mlp.fit(X_train_s, y_train)
        acc_mlp = clf_mlp.score(X_test_s, y_test)

        # Entrenar Árbol de decisión
        clf_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf_tree.fit(X_train_s, y_train)
        acc_tree = clf_tree.score(X_test_s, y_test)

        self._reset_modelo()
        self.scaler = scaler
        self.modelo_mlp = clf_mlp
        self.modelo_tree = clf_tree
        self.modelo_entrenado = True
        return True, f"Entrenado. Acc MLP: {acc_mlp:.2f} | Acc Tree: {acc_tree:.2f}"

    def decision_auto(self) -> int:
        if not self.modelo_entrenado:
            return 0
        if not self.bala_disparada:
            return 0
        
        distancia = abs(self.jugador.x - self.bala.x)

        if self.clase_unica is not None:
            self.ultima_accion_auto = self.clase_unica
            return self.clase_unica

        if self.scaler is None:
            return 0

        X = [[float(self.velocidad_bala), float(distancia), float(self.bala.y)]]
        Xs = self.scaler.transform(X)
        
        if self.usar_tree and self.modelo_tree is not None:
            pred = int(self.modelo_tree.predict(Xs)[0])
        elif not self.usar_tree and self.modelo_mlp is not None:
            pred = int(self.modelo_mlp.predict(Xs)[0])
        else:
            pred = 0
            
        self.ultima_accion_auto = pred
        return pred

    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)
        titulo = self.fuente.render("MENÚ", True, self.BLANCO)
        self.pantalla.blit(titulo, (self.w // 2 - titulo.get_width() // 2, int(60 * self.scale)))

        opciones = [
            "M - Manual (reinicia dataset y borra modelo)",
            "A - Auto (usa modelo; sin modelo NO hace nada)",
            "L - Cambiar Modelo Auto (MLP / Tree)",
            "T - Entrenar Modelos",
            "C - Exportar datos a CSV",
            "F - Fullscreen (toggle)",
            "Q - Salir",
        ]
        x0 = int(80 * self.scale)
        y = int(140 * self.scale)
        line_h = self.fuente.get_linesize()
        pad = max(6, int(6 * self.scale))
        for op in opciones:
            t = self.fuente.render(op, True, self.BLANCO)
            self.pantalla.blit(t, (x0, y))
            y += line_h + pad

        y += int(8 * self.scale)
        modelo_activo = "Árbol de Decisión" if self.usar_tree else "MLP (Red Neuronal)"
        estado = [
            f"Memoria: {len(self.datos_modelo)} | Modelos: {'sí' if self.modelo_entrenado else 'no'}",
            f"Modelo Activo en Auto: {modelo_activo}",
        ]
        for line in estado:
            t = self.fuente_chica.render(line, True, self.GRIS)
            self.pantalla.blit(t, (x0, y))
            y += self.fuente_chica.get_linesize()

        if msg:
            mm = self.fuente_chica.render(msg, True, self.AMARILLO)
            self.pantalla.blit(mm, (x0, y + int(12 * self.scale)))

        pygame.display.flip()

    def mostrar_menu(self) -> None:
        msg = ""
        esperando = True
        self._decision_frame_counter = 0
        while esperando and self.corriendo:
            self._dibujar_menu(msg)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                    esperando = False
                    break
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_m:
                        self.modo_auto = False
                        self.datos_modelo.clear()
                        self._reset_modelo()
                        self._reset_estado_juego()
                        esperando = False
                        break
                    if e.key == pygame.K_a:
                        if not self.modelo_entrenado:
                            msg = "Primero entrena los modelos (T) en esta sesión."
                        else:
                            self.modo_auto = True
                            self._reset_estado_juego()
                            esperando = False
                            break
                    if e.key == pygame.K_l:
                        self.usar_tree = not self.usar_tree
                    if e.key == pygame.K_t:
                        ok, info = self.entrenar_modelo()
                        msg = info if ok else f"Error: {info}"
                    if e.key == pygame.K_c:
                        msg = self.exportar_datos_csv()
                    if e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    if e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False
                        return

    def _update_frame(self) -> None:
        self.fondo_x1 -= self.fondo_speed
        self.fondo_x2 -= self.fondo_speed
        if self.fondo_x1 <= -self.w:
            self.fondo_x1 = self.w
        if self.fondo_x2 <= -self.w:
            self.fondo_x2 = self.w
        self.pantalla.blit(self.fondo_img, (self.fondo_x1, 0))
        self.pantalla.blit(self.fondo_img, (self.fondo_x2, 0))

        self.frame_count += 1
        if self.frame_count >= self.frame_speed:
            self.current_frame = (self.current_frame + 1) % len(self.jugador_frames)
            self.frame_count = 0

        # Dibujar jugador (escalar a la mitad si está agachado)
        frame_actual = self.jugador_frames[self.current_frame]
        if self.agachado:
            img_agachada = pygame.transform.scale(frame_actual, (self.player_size[0], self.player_size[1] // 2))
            self.pantalla.blit(img_agachada, (self.jugador.x, self.jugador.y))
        else:
            self.pantalla.blit(frame_actual, (self.jugador.x, self.jugador.y))
            
        self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))

        if self.bala_disparada:
            self.bala.x += self.velocidad_bala
        if self.bala.x < -self.bullet_size[0]:
            self.reset_bala()
        self.pantalla.blit(self.bala_img, (self.bala.x, self.bala.y))

        if self.jugador.colliderect(self.bala):
            self._reset_estado_juego()

        if self.modelo_entrenado and self.modo_auto and self.ultima_accion_auto is not None:
            txt_accion = {0: "Nada", 1: "Salto", 2: "Agacharse"}.get(self.ultima_accion_auto, "Nada")
            modelo_str = "Tree" if self.usar_tree else "MLP"
            txt = self.fuente_chica.render(f"Acción ({modelo_str}): {txt_accion}", True, self.AMARILLO)
            self.pantalla.blit(txt, (10, 10))

    def loop(self) -> None:
        reloj = pygame.time.Clock()
        self.mostrar_menu()

        while self.corriendo:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        self.corriendo = False
                    elif e.key in (pygame.K_ESCAPE, pygame.K_p):
                        self._reset_estado_juego()
                        self.mostrar_menu()
                    elif e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    elif e.key == pygame.K_SPACE and (not self.modo_auto) and self.en_suelo and not self.agachado:
                        self.iniciar_salto()
                    elif e.key == pygame.K_DOWN and (not self.modo_auto) and self.en_suelo:
                        self.iniciar_agacharse()
                elif e.type == pygame.KEYUP:
                    if e.key == pygame.K_DOWN and not self.modo_auto:
                        self.levantar()

            if not self.corriendo:
                break

            if self.modo_auto:
                accion = self.decision_auto()
                if accion == 1:
                    if self.agachado:
                        self.levantar()
                    self.iniciar_salto()
                elif accion == 2:
                    self.iniciar_agacharse()
                else:
                    if self.agachado:
                        self.levantar()
            else:
                self.registrar_decision_manual()

            if self.salto:
                self.manejar_salto()

            if not self.bala_disparada:
                self.disparar_bala()

            self._update_frame()
            pygame.display.flip()
            reloj.tick(45)

        pygame.quit()

def main() -> None:
    Juego().loop()

if __name__ == "__main__":
    main()

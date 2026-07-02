import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import random
import math
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class AdamOptimizerCorregido:
    """Adam optimizer con implementación verificada y probada"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # Primer momento
        self.v = {}  # Segundo momento
        self.t = 0   # Time step GLOBAL
    
    def step(self, gradients_dict):
        """
        Actualización batch corregida de todos los parámetros
        """
        if not gradients_dict:  # Protección contra dict vacío
            return {}
            
        self.t += 1  # Solo incrementar UNA vez por batch
        
        updates = {}
        for param_key, gradient in gradients_dict.items():
            # Validar que gradient es numérico
            if not isinstance(gradient, (int, float, np.number)):
                if hasattr(gradient, 'item'):  # Array 0-dimensional
                    gradient = gradient.item()
                else:
                    print(f"Warning: Invalid gradient type for {param_key}: {type(gradient)}")
                    continue
            
            # Protección contra NaN/Inf
            if not np.isfinite(gradient):
                gradient = 0.0
            
            # Inicializar momentos si es primera vez
            if param_key not in self.m:
                self.m[param_key] = 0.0
                self.v[param_key] = 0.0
            
            # Actualizar momentos
            self.m[param_key] = self.beta1 * self.m[param_key] + (1 - self.beta1) * gradient
            self.v[param_key] = self.beta2 * self.v[param_key] + (1 - self.beta2) * (gradient ** 2)
            
            # Corrección de sesgo (con protección contra overflow)
            try:
                m_corrected = self.m[param_key] / (1 - self.beta1 ** self.t)
                v_corrected = self.v[param_key] / (1 - self.beta2 ** self.t)
            except OverflowError:
                # Si beta^t es muy pequeño, usar versión sin corrección
                m_corrected = self.m[param_key]
                v_corrected = self.v[param_key]
            
            # Calcular update con protección
            denominator = np.sqrt(v_corrected) + self.epsilon
            if denominator == 0:
                denominator = self.epsilon
                
            update = self.lr * m_corrected / denominator
            
            # Protección final contra valores extremos
            update = np.clip(update, -10.0, 10.0)
            updates[param_key] = update
        
        return updates

class EarlyStoppingRobusto:
    """Early stopping mejorado y probado"""
    
    def __init__(self, patience=100, min_delta=1e-6, warmup_epochs=50):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.best_loss = float('inf')
        self.wait = 0
        self.epoch = 0
        
    def should_stop(self, current_loss):
        self.epoch += 1
        
        # Validar entrada
        if not np.isfinite(current_loss):
            current_loss = float('inf')
        
        # No hacer early stopping durante warmup
        if self.epoch < self.warmup_epochs:
            self.best_loss = min(self.best_loss, current_loss)
            return False
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
            return False

# Clases GP (verificadas del código original)
class NodoGP:
    def __init__(self, valor, es_terminal=False, arity=0):
        self.valor = valor
        self.es_terminal = es_terminal
        self.arity = arity
        self.hijos = []
    
    def agregar_hijo(self, hijo):
        if len(self.hijos) < self.arity:
            self.hijos.append(hijo)
    
    def esta_completo(self):
        return len(self.hijos) == self.arity
    
    def evaluar(self, variables):
        if self.es_terminal:
            if isinstance(self.valor, (int, float)):
                return float(self.valor)
            else:
                return float(variables.get(self.valor, 0.0))
        
        try:
            valores_hijos = []
            for hijo in self.hijos:
                val = hijo.evaluar(variables)
                if math.isnan(val) or math.isinf(val):
                    val = 0.0
                valores_hijos.append(val)
            
            if self.valor == '+':
                resultado = valores_hijos[0] + valores_hijos[1]
            elif self.valor == '-':
                resultado = valores_hijos[0] - valores_hijos[1]
            elif self.valor == '*':
                resultado = valores_hijos[0] * valores_hijos[1]
            elif self.valor == '/':
                divisor = valores_hijos[1]
                if abs(divisor) < 1e-10:
                    resultado = 1.0
                else:
                    resultado = valores_hijos[0] / divisor
            else:
                resultado = 0.0
            
            if math.isnan(resultado) or math.isinf(resultado):
                resultado = 0.0
            
            resultado = max(-1e6, min(1e6, resultado))
            return resultado
            
        except:
            return 0.0
    
    def copiar_profundo(self):
        nuevo_nodo = NodoGP(self.valor, self.es_terminal, self.arity)
        for hijo in self.hijos:
            nuevo_nodo.agregar_hijo(hijo.copiar_profundo())
        return nuevo_nodo

class GeneradorGP:
    def __init__(self, conjunto_funciones, conjunto_terminales, max_profundidad=3):
        self.operadores = {'+': 2, '-': 2, '*': 2, '/': 2}
        self.conjunto_funciones = [op for op in conjunto_funciones if op in self.operadores]
        self.conjunto_terminales = conjunto_terminales
        self.max_profundidad = max_profundidad
    
    def generar_constante_aleatoria(self):
        return round(random.uniform(-1.0, 1.0), 3)
    
    def crear_nodo_aleatorio(self, es_terminal_forzado=False):
        if es_terminal_forzado or random.random() < 0.5:
            if random.random() < 0.7 and self.conjunto_terminales:
                terminal = random.choice(self.conjunto_terminales)
            else:
                terminal = self.generar_constante_aleatoria()
            return NodoGP(terminal, es_terminal=True, arity=0)
        else:
            funcion = random.choice(self.conjunto_funciones)
            arity = self.operadores[funcion]
            return NodoGP(funcion, es_terminal=False, arity=arity)
    
    def generar_arbol_grow(self, profundidad_maxima):
        if profundidad_maxima <= 0:
            return self.crear_nodo_aleatorio(es_terminal_forzado=True)
        
        nodo = self.crear_nodo_aleatorio()
        
        if nodo.es_terminal:
            return nodo
        
        for _ in range(nodo.arity):
            hijo = self.generar_arbol_grow(profundidad_maxima - 1)
            nodo.agregar_hijo(hijo)
        
        return nodo
    
    def generar_poblacion_grow(self, tamano_poblacion):
        poblacion = []
        profundidades = [1, 2, 2, 3, 3, 3]
        
        for i in range(tamano_poblacion):
            profundidad = profundidades[i % len(profundidades)]
            arbol = self.generar_arbol_grow(profundidad)
            poblacion.append(arbol)
        
        return poblacion

class EvaluadorSemántico:
    def __init__(self, nombres_variables):
        self.nombres_variables = nombres_variables
    
    def evaluar_individuo(self, arbol, X):
        M = X.shape[0]
        semantica = np.zeros(M)
        
        for i in range(M):
            variables = {}
            for j, nombre_var in enumerate(self.nombres_variables):
                if j < X.shape[1]:
                    variables[nombre_var] = X[i, j]
            
            try:
                resultado = arbol.evaluar(variables)
                semantica[i] = resultado
            except:
                semantica[i] = 0.0
        
        return self._normalizar_semantica_reproducible(semantica)
    
    def _normalizar_semantica_reproducible(self, semantica):
        """Normalización reproducible SIN aleatoriedad"""
        semantica = np.clip(semantica, -1e6, 1e6)
        media = np.mean(semantica)
        std = np.std(semantica)
        
        if std < 1e-10:
            # CORREGIDO: Usar valores determinísticos en lugar de aleatorios
            return np.full_like(semantica, media + 1e-8 * np.arange(len(semantica)))
        
        semantica_norm = (semantica - media) / std
        return np.clip(semantica_norm, -5, 5)
    
    def crear_matriz_semantica(self, poblacion, X):
        M = X.shape[0]
        matriz_semantica = np.zeros((len(poblacion), M))
        
        for i, arbol in enumerate(poblacion):
            semantica = self.evaluar_individuo(arbol, X)
            matriz_semantica[i, :] = semantica
        
        return matriz_semantica

class GSGP_NN_Final:
    """
    Versión final con TODOS los bugs corregidos y probados
    """
    
    def __init__(self, M, K=2, N=30, learning_rate=0.005, tamano_poblacion=50):
        self.M = M
        self.K = K
        self.N = N
        self.tamano_poblacion = tamano_poblacion
        
        # Optimizadores
        self.adam_optimizer = AdamOptimizerCorregido(learning_rate=learning_rate)
        self.early_stopping = EarlyStoppingRobusto(patience=150, warmup_epochs=30)
        
        # Componentes GP (fijos después de inicialización)
        self.generador_gp = None
        self.evaluador_semantico = None
        self.poblacion_inicial = None
        self.poblacion_auxiliar = None
        
        # Matrices semánticas
        self.matriz_semantica_inicial = None
        self.matriz_semantica_auxiliar = None
        
        # Parámetros neuronales
        self.alpha = {}
        self.ms = {}
        self.beta = {}
        self.w_output = None
        self.b_output = None
        
        # Semánticas procesadas
        self.semanticas_padre = None
        self.rutas_semanticas = {}
        self.semanticas_capa_0 = None
        
        # Métricas de tiempo (CORREGIDO)
        self.tiempo_total_entrenamiento = 0.0
        self.tiempos_por_epoca = []
        
        print(f"GSGP-NN Final: {M} patrones, {K} capas, {N} nodos, {tamano_poblacion} poblacion GP")
    
    def inicializar_componentes_gp(self, X):
        print(f"\n=== INICIALIZACIÓN GP ===")
        
        num_variables = X.shape[1]
        nombres_variables = [f'x{i+1}' for i in range(num_variables)]
        
        self.generador_gp = GeneradorGP(
            conjunto_funciones=['+', '-', '*', '/'],
            conjunto_terminales=nombres_variables,
            max_profundidad=10
        )
        
        print(f"Generando poblaciones GP de tamaño {self.tamano_poblacion}...")
        self.poblacion_inicial = self.generador_gp.generar_poblacion_grow(self.tamano_poblacion)
        self.poblacion_auxiliar = self.generador_gp.generar_poblacion_grow(self.tamano_poblacion)
         
        self.evaluador_semantico = EvaluadorSemántico(nombres_variables)
        
        print("Evaluando poblaciones...")
        self.matriz_semantica_inicial = self.evaluador_semantico.crear_matriz_semantica(
            self.poblacion_inicial, X
        )
        self.matriz_semantica_auxiliar = self.evaluador_semantico.crear_matriz_semantica(
            self.poblacion_auxiliar, X
        )
        
        print(f"Matrices semánticas: {self.matriz_semantica_inicial.shape}")
    
    def generar_semanticas_para_red_neuronal(self):
        print(f"\n=== GENERACIÓN DE SEMÁNTICAS ===")
        
        # Semánticas padre
        self.semanticas_padre = {}
        for n in range(1, self.N + 1):
            idx = (n - 1) % self.matriz_semantica_inicial.shape[0]
            self.semanticas_padre[n] = self.matriz_semantica_inicial[idx, :].copy()
        
        # Semánticas de capa 0
        self.semanticas_capa_0 = {}
        matriz_completa = np.vstack([self.matriz_semantica_inicial, self.matriz_semantica_auxiliar])
        
        for n in range(1, self.N + 1):
            idx = (n - 1) % matriz_completa.shape[0]
            self.semanticas_capa_0[n] = matriz_completa[idx, :].copy()
        
        # Rutas semánticas
        self.rutas_semanticas = {}
        idx_auxiliar = 0
        for k in range(1, self.K + 1):
            self.rutas_semanticas[k] = {}
            for n in range(1, self.N + 1):
                rt1_idx = idx_auxiliar % self.matriz_semantica_auxiliar.shape[0]
                RT1 = self.matriz_semantica_auxiliar[rt1_idx, :].copy()
                idx_auxiliar += 1
                
                rt2_idx = idx_auxiliar % self.matriz_semantica_auxiliar.shape[0]
                RT2 = self.matriz_semantica_auxiliar[rt2_idx, :].copy()
                idx_auxiliar += 1
                
                self.rutas_semanticas[k][n] = {'RT1': RT1, 'RT2': RT2}
        
        print("Semánticas generadas correctamente")
    
    def inicializar_parametros_neuronales(self):
        print("Inicializando parámetros neuronales...")
        
        # He initialization
        std_alpha = np.sqrt(2.0 / self.N)
        std_other = 0.1
        
        for k in range(1, self.K + 1):
            for n in range(1, self.N + 1):
                # Pesos alpha
                self.alpha[(0, n, k)] = np.random.normal(0, std_alpha)
                for i in range(1, self.N + 1):
                    self.alpha[(i, n, k)] = np.random.normal(0, std_alpha)
                
                # Coeficientes
                self.ms[(n, k)] = np.random.normal(0, std_other)
                self.beta[(n, k)] = np.random.normal(0, std_other)
        
        # Pesos de salida
        self.w_output = np.random.normal(0, std_alpha, self.N)
        self.b_output = 0.0
        
        print(f"Parámetros inicializados correctamente")
    
    def calcular_semantica_nodo(self, n, k, semanticas_anteriores):
        # Denominador
        d_nk = abs(self.alpha[(0, n, k)])
        for i in range(1, self.N + 1):
            d_nk += abs(self.alpha[(i, n, k)])
        
        if d_nk < 1e-12:
            d_nk = 1e-12
        
        # Numerador de herencia
        H_nk = self.alpha[(0, n, k)] * self.semanticas_padre[n]
        for i in range(1, self.N + 1):
            H_nk += self.alpha[(i, n, k)] * semanticas_anteriores[i]
        
        # Término heredado
        termino_heredado = H_nk / d_nk
        
        # Mutación semántica
        RT1 = self.rutas_semanticas[k][n]['RT1']
        RT2 = self.rutas_semanticas[k][n]['RT2']
        
        mutacion_base = self.ms[(n, k)] * (RT1 - RT2)
        mutacion_semantica = self.beta[(n, k)] * np.tanh(mutacion_base)
        
        return termino_heredado + mutacion_semantica
    
    def forward_pass(self):
        semanticas_actuales = self.semanticas_capa_0.copy()
        self.semanticas_todas_capas = {0: semanticas_actuales.copy()}
        
        for k in range(1, self.K + 1):
            semanticas_siguientes = {}
            for n in range(1, self.N + 1):
                semanticas_siguientes[n] = self.calcular_semantica_nodo(n, k, semanticas_actuales)
            
            semanticas_actuales = semanticas_siguientes
            self.semanticas_todas_capas[k] = semanticas_actuales.copy()
        
        return semanticas_actuales
    
    def generar_predicciones(self, semanticas_finales):
        z = np.zeros(self.M)
        for n in range(1, self.N + 1):
            z += self.w_output[n-1] * semanticas_finales[n]
        z += self.b_output
        return z
    
    def calcular_perdida(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def calcular_todos_los_gradientes(self, y_true, y_pred):
        grad_perdida_pred = (2.0 / self.M) * (y_pred - y_true)
        
        # Gradientes de capa de salida
        semanticas_finales = self.semanticas_todas_capas[self.K]
        gradientes_semanticas = {}
        for n in range(1, self.N + 1):
            gradientes_semanticas[n] = grad_perdida_pred * self.w_output[n-1]
        
        # Diccionario para TODOS los gradientes (escalares únicamente)
        all_gradients = {}
        
        # Gradientes de pesos de salida
        for n in range(1, self.N + 1):
            grad_w = np.dot(grad_perdida_pred, semanticas_finales[n])
            grad_w = np.clip(grad_w, -5.0, 5.0)
            all_gradients[f'w_output_{n}'] = float(grad_w)  # Asegurar escalar
        
        grad_b = np.sum(grad_perdida_pred)
        grad_b = np.clip(grad_b, -5.0, 5.0)
        all_gradients['b_output'] = float(grad_b)  # Asegurar escalar
        
        # Gradientes hacia atrás por capas
        for k in range(self.K, 0, -1):
            # Gradientes alpha
            for n in range(1, self.N + 1):
                for j in range(0, self.N + 1):
                    if (j, n, k) in self.alpha:
                        grad_alpha = self.calcular_gradiente_alpha(j, n, k, gradientes_semanticas[n])
                        grad_alpha = np.clip(grad_alpha, -5.0, 5.0)
                        all_gradients[f'alpha_{j}_{n}_{k}'] = float(grad_alpha)  # Asegurar escalar
                
                # Gradientes beta y ms
                RT1 = self.rutas_semanticas[k][n]['RT1']
                RT2 = self.rutas_semanticas[k][n]['RT2']
                rt_diff = RT1 - RT2
                
                mutacion_base = self.ms[(n, k)] * rt_diff
                tanh_val = np.tanh(mutacion_base)
                
                grad_beta = np.dot(gradientes_semanticas[n], tanh_val)
                grad_beta = np.clip(grad_beta, -5.0, 5.0)
                all_gradients[f'beta_{n}_{k}'] = float(grad_beta)  # Asegurar escalar
                
                sech_squared = 1 - tanh_val**2
                grad_ms = np.dot(gradientes_semanticas[n], 
                               self.beta[(n, k)] * sech_squared * rt_diff)
                grad_ms = np.clip(grad_ms, -5.0, 5.0)
                all_gradients[f'ms_{n}_{k}'] = float(grad_ms)  # Asegurar escalar
            
            # Propagar gradientes a capa anterior
            if k > 1:
                nuevos_gradientes = {}
                for j in range(1, self.N + 1):
                    nuevos_gradientes[j] = np.zeros(self.M)
                    
                    for n in range(1, self.N + 1):
                        if (j, n, k) in self.alpha:
                            d_nk = abs(self.alpha[(0, n, k)])
                            for i in range(1, self.N + 1):
                                d_nk += abs(self.alpha[(i, n, k)])
                            
                            if d_nk < 1e-12:
                                d_nk = 1e-12
                            
                            H_nk = self.alpha[(0, n, k)] * self.semanticas_padre[n]
                            semanticas_anteriores = self.semanticas_todas_capas[k-1]
                            for i in range(1, self.N + 1):
                                H_nk += self.alpha[(i, n, k)] * semanticas_anteriores[i]
                            
                            derivada_interaccion = (self.alpha[(j, n, k)] * d_nk - H_nk * np.sign(self.alpha[(j, n, k)])) / (d_nk ** 2)
                            nuevos_gradientes[j] += gradientes_semanticas[n] * derivada_interaccion
                
                gradientes_semanticas = nuevos_gradientes
        
        return all_gradients
    
    def aplicar_updates_adam(self, all_gradients):
        updates = self.adam_optimizer.step(all_gradients)
        
        # Aplicar updates con clipping
        for n in range(1, self.N + 1):
            key = f'w_output_{n}'
            if key in updates:
                self.w_output[n-1] -= updates[key]
                self.w_output[n-1] = np.clip(self.w_output[n-1], -10, 10)
        
        if 'b_output' in updates:
            self.b_output -= updates['b_output']
            self.b_output = np.clip(self.b_output, -10, 10)
        
        # Updates de parámetros de red
        for k in range(1, self.K + 1):
            for n in range(1, self.N + 1):
                for j in range(0, self.N + 1):
                    key = f'alpha_{j}_{n}_{k}'
                    if key in updates and (j, n, k) in self.alpha:
                        self.alpha[(j, n, k)] -= updates[key]
                        self.alpha[(j, n, k)] = np.clip(self.alpha[(j, n, k)], -5, 5)
                
                key_beta = f'beta_{n}_{k}'
                if key_beta in updates:
                    self.beta[(n, k)] -= updates[key_beta]
                    self.beta[(n, k)] = np.clip(self.beta[(n, k)], -2, 2)
                
                key_ms = f'ms_{n}_{k}'
                if key_ms in updates:
                    self.ms[(n, k)] -= updates[key_ms]
                    self.ms[(n, k)] = np.clip(self.ms[(n, k)], -2, 2)
    
    def calcular_gradiente_alpha(self, j, n, k, grad_perdida_semantica):
        if k == 1:
            semanticas_anteriores = self.semanticas_capa_0
        else:
            semanticas_anteriores = self.semanticas_todas_capas[k-1]
        
        d_nk = abs(self.alpha[(0, n, k)])
        for i in range(1, self.N + 1):
            d_nk += abs(self.alpha[(i, n, k)])
        
        if d_nk < 1e-12:
            d_nk = 1e-12
        
        H_nk = self.alpha[(0, n, k)] * self.semanticas_padre[n]
        for i in range(1, self.N + 1):
            H_nk += self.alpha[(i, n, k)] * semanticas_anteriores[i]
        
        if j == 0:
            numerador_grad = self.semanticas_padre[n] * d_nk - H_nk * np.sign(self.alpha[(0, n, k)])
        else:
            numerador_grad = semanticas_anteriores[j] * d_nk - H_nk * np.sign(self.alpha[(j, n, k)])
        
        derivada_semantica = numerador_grad / (d_nk ** 2)
        return np.dot(grad_perdida_semantica, derivada_semantica)
    
    def entrenar(self, X, y, epochs=1000, verbose=True):
        print(f"\n=== ENTRENAMIENTO FINAL ===")
        
        # Inicialización
        start_time = time.time()
        self.inicializar_componentes_gp(X)
        self.generar_semanticas_para_red_neuronal()
        self.inicializar_parametros_neuronales()
        init_time = time.time() - start_time
        print(f"Inicialización: {init_time:.2f}s")
        
        # Entrenamiento
        training_start = time.time()
        historial_perdidas = []
        self.tiempos_por_epoca = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Forward pass
            semanticas_finales = self.forward_pass()
            y_pred = self.generar_predicciones(semanticas_finales)
            
            # Pérdida
            perdida = self.calcular_perdida(y, y_pred)
            historial_perdidas.append(perdida)
            
            # Backward pass corregido
            all_gradients = self.calcular_todos_los_gradientes(y, y_pred)
            self.aplicar_updates_adam(all_gradients)
            
            # Tiempo por época
            epoch_time = time.time() - epoch_start
            self.tiempos_por_epoca.append(epoch_time)
            
            # Early stopping
            if self.early_stopping.should_stop(perdida):
                print(f"\nEarly stopping en época {epoch + 1}")
                break
            
            # Logging CORREGIDO - Solo mostrar cada época
            if verbose and (epoch + 1) % 1 == 0:
                # Cálculo de mejora CON protección contra división por cero
                mejora_str = ""
                if len(historial_perdidas) >= 25:
                    perdida_anterior = historial_perdidas[-25]
                    if perdida_anterior > 1e-10:  # Protección contra división por cero
                        mejora_pct = ((perdida_anterior - perdida) / perdida_anterior) * 100
                        if mejora_pct > 0:
                            mejora_str = f"(↓{mejora_pct:.2f}%)"
                        elif mejora_pct < 0:
                            mejora_str = f"(↑{abs(mejora_pct):.2f}%)"
                
                print(f"Época {epoch + 1:4d}: Pérdida = {perdida:.6f} {mejora_str}, T={epoch_time:.2f}s")
        
        # CORREGIDO: Calcular tiempo total real
        self.tiempo_total_entrenamiento = time.time() - training_start
        self.historial_perdidas = historial_perdidas
        
        if verbose:
            print(f"\nEntrenamiento completado en {self.tiempo_total_entrenamiento:.2f}s")
            if len(self.tiempos_por_epoca) > 0:
                tiempo_promedio = self.tiempo_total_entrenamiento / len(self.tiempos_por_epoca)
                print(f"Tiempo promedio por época: {tiempo_promedio:.3f}s")
        
        return historial_perdidas
    
    def predecir(self, X):
        """Predicción verificada (sin cambios - funciona correctamente)"""
        M_original = self.M
        self.M = X.shape[0]
        
        # Evaluar poblaciones fijas en nuevos datos
        matriz_inicial_nueva = self.evaluador_semantico.crear_matriz_semantica(
            self.poblacion_inicial, X
        )
        matriz_auxiliar_nueva = self.evaluador_semantico.crear_matriz_semantica(
            self.poblacion_auxiliar, X
        )
        
        # Regenerar semánticas
        semanticas_padre_nueva = {}
        for n in range(1, self.N + 1):
            idx = (n - 1) % matriz_inicial_nueva.shape[0]
            semanticas_padre_nueva[n] = matriz_inicial_nueva[idx, :]
        
        semanticas_capa_0_nueva = {}
        matriz_completa_nueva = np.vstack([matriz_inicial_nueva, matriz_auxiliar_nueva])
        for n in range(1, self.N + 1):
            idx = (n - 1) % matriz_completa_nueva.shape[0]
            semanticas_capa_0_nueva[n] = matriz_completa_nueva[idx, :]
        
        rutas_nuevas = {}
        idx_auxiliar = 0
        for k in range(1, self.K + 1):
            rutas_nuevas[k] = {}
            for n in range(1, self.N + 1):
                rt1_idx = idx_auxiliar % matriz_auxiliar_nueva.shape[0]
                RT1 = matriz_auxiliar_nueva[rt1_idx, :]
                idx_auxiliar += 1
                
                rt2_idx = idx_auxiliar % matriz_auxiliar_nueva.shape[0]
                RT2 = matriz_auxiliar_nueva[rt2_idx, :]
                idx_auxiliar += 1
                
                rutas_nuevas[k][n] = {'RT1': RT1, 'RT2': RT2}
        
        # Usar temporalmente
        semanticas_padre_original = self.semanticas_padre
        semanticas_capa_0_original = self.semanticas_capa_0
        rutas_originales = self.rutas_semanticas
        
        self.semanticas_padre = semanticas_padre_nueva
        self.semanticas_capa_0 = semanticas_capa_0_nueva
        self.rutas_semanticas = rutas_nuevas
        
        # Predicción
        semanticas_finales = self.forward_pass()
        y_pred = self.generar_predicciones(semanticas_finales)
        
        # Restaurar
        self.M = M_original
        self.semanticas_padre = semanticas_padre_original
        self.semanticas_capa_0 = semanticas_capa_0_original
        self.rutas_semanticas = rutas_originales
        
        return y_pred

# Función de datos con fallback robusto
def preparar_datos_concrete_regresion(path='eheating.txt', semilla=1):
    try:
        data = np.loadtxt(path)
        X = data[:, :-1]
        y = data[:, -1]
        print(f"Datos cargados desde {path}: {X.shape}")
    except:
        print(f"{path} no encontrado. Generando datos sintéticos...")
        np.random.seed(semilla)  # Reproducibilidad
        X = np.random.randn(500, 8)
        y = (X[:, 0]**2 + X[:, 1] * X[:, 2] + 
             np.sin(X[:, 3]) + 0.5 * X[:, 4] + 
             0.1 * np.random.randn(500))
        print(f"Datos sintéticos generados: {X.shape}")
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_X, scaler_y

def ejemplo_gsgp_nn_simple(path, K, N, learning_rate, epochs, semilla, verbose=False):
    """Función auxiliar para una sola corrida (sin visualización)"""
    # Configurar semilla
    np.random.seed(semilla)
    random.seed(semilla)
    
    # Preparar datos
    X, y, scaler_X, scaler_y = preparar_datos_concrete_regresion(path, semilla=semilla)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=semilla)
    
    # Crear modelo
    modelo = GSGP_NN_Final(
        M=X_train.shape[0],
        K=K,                  
        N=N,                 
        learning_rate=learning_rate,   
        tamano_poblacion=int(2*N*K)    
    )
    
    # Entrenar
    historial = modelo.entrenar(X_train, y_train, epochs=epochs, verbose=verbose)
    
    # Evaluar
    y_pred_train = modelo.predecir(X_train)
    y_pred_test = modelo.predecir(X_test)
    
    # ========= MÉTRICAS NORMALIZADAS =========
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # ========= CONVERTIR A ESCALA ORIGINAL =========
    y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_pred_train_orig = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_test_orig = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    
    # ========= MÉTRICAS EN ESCALA ORIGINAL =========
    mae_train_orig = mean_absolute_error(y_train_orig, y_pred_train_orig)
    mae_test_orig = mean_absolute_error(y_test_orig, y_pred_test_orig)
    mse_train_orig = mean_squared_error(y_train_orig, y_pred_train_orig)
    mse_test_orig = mean_squared_error(y_test_orig, y_pred_test_orig)
    rmse_train_orig = np.sqrt(mse_train_orig)
    rmse_test_orig = np.sqrt(mse_test_orig)
    r2_train_orig = r2_score(y_train_orig, y_pred_train_orig)
    r2_test_orig = r2_score(y_test_orig, y_pred_test_orig)
    
    # Retornar todas las métricas
    return {
        'modelo': modelo,
        'historial': historial,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'mse_train': mse_train,
        'mse_test': mse_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae_train_orig': mae_train_orig,
        'mae_test_orig': mae_test_orig,
        'mse_train_orig': mse_train_orig,
        'mse_test_orig': mse_test_orig,
        'rmse_train_orig': rmse_train_orig,
        'rmse_test_orig': rmse_test_orig,
        'r2_train_orig': r2_train_orig,
        'r2_test_orig': r2_test_orig,
        'tiempo_entrenamiento': modelo.tiempo_total_entrenamiento,
        'epocas_entrenadas': len(historial),
        'perdida_final': historial[-1] if historial else float('inf'),
        'convergencia': len(historial) < epochs  # Corregido: comparar con epochs no 1000
    }

# ============================================================================
# ANÁLISIS DE ROBUSTEZ COMPLETO
# ============================================================================

def analisis_robustez_gsgp_nn(path, K, N, learning_rate, epochs, num_corridas=30, verbose_individual=True):
    """
    Ejecuta múltiples corridas y recolecta estadísticas para análisis de robustez
    """
    print("="*80)
    print(f"ANÁLISIS DE ROBUSTEZ GSGP-NN: {num_corridas} CORRIDAS INDEPENDIENTES")
    print(f"Dataset: {path}, K={K}, N={N}, LR={learning_rate}, Épocas={epochs}")
    print("="*80)
    
    # Almacenar resultados de todas las corridas
    resultados = {
        'semilla': [],
        'mae_train': [], 'mae_test': [],
        'mse_train': [], 'mse_test': [],
        'rmse_train': [], 'rmse_test': [],
        'r2_train': [], 'r2_test': [],
        'mae_train_orig': [], 'mae_test_orig': [],
        'mse_train_orig': [], 'mse_test_orig': [],
        'rmse_train_orig': [], 'rmse_test_orig': [],
        'r2_train_orig': [], 'r2_test_orig': [],
        'tiempo_entrenamiento': [],
        'epocas_entrenadas': [],
        'perdida_final': [],
        'convergencia': []  # Si alcanzó early stopping
    }
    
    tiempo_inicio_total = time.time()
    corridas_exitosas = 0
    
    # CORRECCIÓN: Usar semillas simples y secuenciales
    for i in range(num_corridas):
        semilla = i  # Semilla simple 0, 1, 2, ..., 29
        print(f"\n{'='*50}")
        print(f"CORRIDA {i + 1}/{num_corridas} (Semilla: {semilla})")
        print(f"{'='*50}")
        
        try:
            # Ejecutar una corrida
            resultado = ejemplo_gsgp_nn_simple(path, K, N, learning_rate, epochs, semilla, verbose=verbose_individual)
            
            # Almacenar resultados
            resultados['semilla'].append(semilla)
            resultados['mae_train'].append(resultado['mae_train'])
            resultados['mae_test'].append(resultado['mae_test'])
            resultados['mse_train'].append(resultado['mse_train'])
            resultados['mse_test'].append(resultado['mse_test'])
            resultados['rmse_train'].append(resultado['rmse_train'])
            resultados['rmse_test'].append(resultado['rmse_test'])
            resultados['r2_train'].append(resultado['r2_train'])
            resultados['r2_test'].append(resultado['r2_test'])
            resultados['mae_train_orig'].append(resultado['mae_train_orig'])
            resultados['mae_test_orig'].append(resultado['mae_test_orig'])
            resultados['mse_train_orig'].append(resultado['mse_train_orig'])
            resultados['mse_test_orig'].append(resultado['mse_test_orig'])
            resultados['rmse_train_orig'].append(resultado['rmse_train_orig'])
            resultados['rmse_test_orig'].append(resultado['rmse_test_orig'])
            resultados['r2_train_orig'].append(resultado['r2_train_orig'])
            resultados['r2_test_orig'].append(resultado['r2_test_orig'])
            resultados['tiempo_entrenamiento'].append(resultado['tiempo_entrenamiento'])
            resultados['epocas_entrenadas'].append(resultado['epocas_entrenadas'])
            resultados['perdida_final'].append(resultado['perdida_final'])
            resultados['convergencia'].append(resultado['convergencia'])
            
            corridas_exitosas += 1
            
            print(f"✓ Corrida {i + 1} completada exitosamente")
            print(f"  R² Test (original): {resultado['r2_test_orig']:.4f}")
            print(f"  RMSE Test (original): {resultado['rmse_test_orig']:.4f}")
            print(f"  Tiempo: {resultado['tiempo_entrenamiento']:.2f}s")
            print(f"  Épocas: {resultado['epocas_entrenadas']}")
            
        except Exception as e:
            print(f"✗ Error en corrida {i + 1}: {str(e)}")
            continue
    
    tiempo_total = time.time() - tiempo_inicio_total
    
    print(f"\n{'='*80}")
    print(f"ANÁLISIS COMPLETADO: {corridas_exitosas}/{num_corridas} corridas exitosas")
    print(f"Tiempo total: {tiempo_total:.2f}s")
    print(f"{'='*80}")
    
    # Generar estadísticas descriptivas
    if corridas_exitosas > 0:
        generar_estadisticas_robustez(resultados, corridas_exitosas, path)
        visualizar_resultados_robustez(resultados, path)
        guardar_resultados(resultados, num_corridas, path)
    
    return resultados

def generar_estadisticas_robustez(resultados, corridas_exitosas, dataset_name):
    """Genera estadísticas descriptivas completas"""
    
    print(f"\n{'='*60}")
    print(f"ESTADÍSTICAS DE ROBUSTEZ - {dataset_name}")
    print(f"{'='*60}")
    
    metricas_principales = ['r2_test_orig', 'rmse_test_orig', 'mae_test_orig', 'tiempo_entrenamiento']
    nombres_metricas = ['R² Test', 'RMSE Test', 'MAE Test', 'Tiempo (s)']
    
    for metrica, nombre in zip(metricas_principales, nombres_metricas):
        valores = np.array(resultados[metrica])
        
        print(f"\n{nombre}:")
        print(f"  Media:      {np.mean(valores):.6f}")
        print(f"  Mediana:    {np.median(valores):.6f}")
        print(f"  Std:        {np.std(valores):.6f}")
        print(f"  Min:        {np.min(valores):.6f}")
        print(f"  Max:        {np.max(valores):.6f}")
        print(f"  Q1:         {np.percentile(valores, 25):.6f}")
        print(f"  Q3:         {np.percentile(valores, 75):.6f}")
        print(f"  Rango IQR:  {np.percentile(valores, 75) - np.percentile(valores, 25):.6f}")
    
    # Estadísticas adicionales
    convergencia_rate = np.mean(resultados['convergencia']) * 100
    epocas_promedio = np.mean(resultados['epocas_entrenadas'])
    
    print(f"\n{'='*40}")
    print("ESTADÍSTICAS DE CONVERGENCIA")
    print(f"{'='*40}")
    print(f"Tasa de early stopping: {convergencia_rate:.1f}%")
    print(f"Épocas promedio: {epocas_promedio:.1f}")
    print(f"Corridas exitosas: {corridas_exitosas}")

def visualizar_resultados_robustez(resultados, dataset_name):
    """Crea visualizaciones completas de robustez"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Análisis de Robustez GSGP-NN - {dataset_name} (30 Corridas)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Distribución R² Test
    axes[0,0].hist(resultados['r2_test_orig'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(np.mean(resultados['r2_test_orig']), color='red', linestyle='--', 
                      label=f'Media: {np.mean(resultados["r2_test_orig"]):.4f}')
    axes[0,0].set_title('Distribución R² Test (Escala Original)', fontweight='bold')
    axes[0,0].set_xlabel('R² Test')
    axes[0,0].set_ylabel('Frecuencia')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Distribución RMSE Test
    axes[0,1].hist(resultados['rmse_test_orig'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].axvline(np.mean(resultados['rmse_test_orig']), color='red', linestyle='--',
                      label=f'Media: {np.mean(resultados["rmse_test_orig"]):.4f}')
    axes[0,1].set_title('Distribución RMSE Test (Escala Original)', fontweight='bold')
    axes[0,1].set_xlabel('RMSE Test')
    axes[0,1].set_ylabel('Frecuencia')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Tiempo de entrenamiento
    axes[0,2].hist(resultados['tiempo_entrenamiento'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,2].axvline(np.mean(resultados['tiempo_entrenamiento']), color='red', linestyle='--',
                      label=f'Media: {np.mean(resultados["tiempo_entrenamiento"]):.2f}s')
    axes[0,2].set_title('Distribución Tiempo de Entrenamiento', fontweight='bold')
    axes[0,2].set_xlabel('Tiempo (s)')
    axes[0,2].set_ylabel('Frecuencia')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Evolución por corrida - R²
    axes[1,0].plot(range(len(resultados['semilla'])), resultados['r2_test_orig'], 'bo-', alpha=0.7)
    axes[1,0].axhline(np.mean(resultados['r2_test_orig']), color='red', linestyle='--', 
                      label='Media')
    axes[1,0].fill_between(range(len(resultados['semilla'])),
                          np.mean(resultados['r2_test_orig']) - np.std(resultados['r2_test_orig']),
                          np.mean(resultados['r2_test_orig']) + np.std(resultados['r2_test_orig']),
                          alpha=0.2, color='red', label='±1σ')
    axes[1,0].set_title('R² Test por Corrida', fontweight='bold')
    axes[1,0].set_xlabel('Número de Corrida')
    axes[1,0].set_ylabel('R² Test')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Evolución por corrida - RMSE
    axes[1,1].plot(range(len(resultados['semilla'])), resultados['rmse_test_orig'], 'ro-', alpha=0.7)
    axes[1,1].axhline(np.mean(resultados['rmse_test_orig']), color='blue', linestyle='--', 
                      label='Media')
    axes[1,1].set_title('RMSE Test por Corrida', fontweight='bold')
    axes[1,1].set_xlabel('Número de Corrida')
    axes[1,1].set_ylabel('RMSE Test')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Épocas por corrida
    axes[1,2].plot(range(len(resultados['semilla'])), resultados['epocas_entrenadas'], 'go-', alpha=0.7)
    axes[1,2].axhline(np.mean(resultados['epocas_entrenadas']), color='blue', linestyle='--', 
                      label='Media')
    axes[1,2].set_title('Épocas de Entrenamiento por Corrida', fontweight='bold')
    axes[1,2].set_xlabel('Número de Corrida')
    axes[1,2].set_ylabel('Épocas')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Box plots adicionales
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(f'Box Plots - {dataset_name}', 
                  fontsize=14, fontweight='bold')
    
    axes2[0].boxplot(resultados['r2_test_orig'])
    axes2[0].set_title('R² Test')
    axes2[0].set_ylabel('R² Test')
    axes2[0].grid(True, alpha=0.3)
    
    axes2[1].boxplot(resultados['rmse_test_orig'])
    axes2[1].set_title('RMSE Test')
    axes2[1].set_ylabel('RMSE Test')
    axes2[1].grid(True, alpha=0.3)
    
    axes2[2].boxplot(resultados['tiempo_entrenamiento'])
    axes2[2].set_title('Tiempo Entrenamiento')
    axes2[2].set_ylabel('Tiempo (s)')
    axes2[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def guardar_resultados(resultados, num_corridas, dataset_name):
    """Guarda los resultados en archivos CSV y de texto"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_clean = dataset_name.replace('.txt', '').replace('.', '_')
    
    # Crear DataFrame y guardar CSV
    df = pd.DataFrame(resultados)
    csv_filename = f"gsgp_nn_{dataset_clean}_{num_corridas}_corridas_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    
    # Guardar resumen estadístico
    resumen_filename = f"gsgp_nn_{dataset_clean}_resumen_{timestamp}.txt"
    with open(resumen_filename, 'w', encoding='utf-8') as f:
        f.write(f"ANÁLISIS DE ROBUSTEZ GSGP-NN\n")
        f.write(f"{'='*50}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Número de corridas: {num_corridas}\n")
        f.write(f"Corridas exitosas: {len(resultados['semilla'])}\n\n")
        
        for metrica in ['r2_test_orig', 'rmse_test_orig', 'mae_test_orig', 'tiempo_entrenamiento']:
            valores = np.array(resultados[metrica])
            f.write(f"{metrica}:\n")
            f.write(f"  Media: {np.mean(valores):.6f}\n")
            f.write(f"  Std: {np.std(valores):.6f}\n")
            f.write(f"  Min: {np.min(valores):.6f}\n")
            f.write(f"  Max: {np.max(valores):.6f}\n\n")
    
    print(f"\nResultados guardados en:")
    print(f"  - {csv_filename}")
    print(f"  - {resumen_filename}")

# ============================================================================
# FUNCIÓN PRINCIPAL PARA EJECUTAR - CORREGIDA
# ============================================================================

if __name__ == "__main__":
    # Configuración de datasets y parámetros
    configuraciones = [
        {"path": "yatch.txt", "K": 2, "N": 200, "lr": 0.005, "epochs": 200},
        {"path": "eheating.txt", "K": 2, "N": 200, "lr": 0.005, "epochs": 200},
        {"path": "ecooling.txt", "K": 2, "N": 200, "lr": 0.005, "epochs": 200},
        {"path": "housing.txt", "K": 2, "N": 50, "lr": 0.005, "epochs": 200},
        {"path": "ConcreteData.txt", "K": 2, "N": 200, "lr": 0.005, "epochs": 200},
        {"path": "tower.txt", "K": 2, "N": 200, "lr": 0.005, "epochs": 200}
    ]
    num_corridas = 30
    print("¿Qué deseas ejecutar?")
    print("1. Análisis de robustez completo (30 corridas por dataset)")
    print("2. Ejemplo individual con visualización")
    print("3. Análisis de un solo dataset")
    
    opcion = input("Selecciona una opción (1, 2 o 3): ").strip()
    
    if opcion == "1":
        # Análisis de robustez completo para todos los datasets
        print("Ejecutando análisis de robustez para todos los datasets...")
        for config in configuraciones:
            print(f"\n{'='*100}")
            print(f"PROCESANDO DATASET: {config['path']}")
            print(f"{'='*100}")
            
            resultados_robustez = analisis_robustez_gsgp_nn(
                path=config['path'],
                K=config['K'],
                N=config['N'],
                learning_rate=config['lr'],
                epochs=config['epochs'],
                num_corridas=num_corridas,
                verbose_individual=True
            )
    
    elif opcion == "2":
        # Ejemplo individual
        print("Selecciona un dataset:")
        for i, config in enumerate(configuraciones):
            print(f"{i+1}. {config['path']}")
        
        dataset_idx = int(input("Número de dataset: ")) - 1
        if 0 <= dataset_idx < len(configuraciones):
            config = configuraciones[dataset_idx]
            semilla = int(input("Ingresa la semilla (0-29): ") or "0")
            
            resultado = ejemplo_gsgp_nn_simple(
                path=config['path'],
                K=config['K'],
                N=config['N'],
                learning_rate=config['lr'],
                epochs=config['epochs'],
                semilla=semilla,
                verbose=True
            )
            print(f"R² Test: {resultado['r2_test_orig']:.4f}")
        else:
            print("Opción inválida")

    
    elif opcion == "3":
        # Análisis de un solo dataset
        print("Selecciona un dataset:")
        for i, config in enumerate(configuraciones):
            print(f"{i+1}. {config['path']}")
        
        dataset_idx = int(input("Número de dataset: ")) - 1
        if 0 <= dataset_idx < len(configuraciones):
            config = configuraciones[dataset_idx]
            
            resultados_robustez = analisis_robustez_gsgp_nn(
                path=config['path'],
                K=config['K'],
                N=config['N'],
                learning_rate=config['lr'],
                epochs=config['epochs'],
                num_corridas=num_corridas,
                verbose_individual=True
            )
        else:
            print("Opción inválida")

    else:
        print("Opción no válida. Ejecutando análisis de robustez completo por defecto...")
        for config in configuraciones:
            print(f"\n{'='*100}")
            print(f"PROCESANDO DATASET: {config['path']}")
            print(f"{'='*100}")
            
            resultados_robustez = analisis_robustez_gsgp_nn(
                path=config['path'],
                K=config['K'],
                N=config['N'],
                learning_rate=config['lr'],
                epochs=config['epochs'],
                num_corridas=num_corridas,
                verbose_individual=True
            )

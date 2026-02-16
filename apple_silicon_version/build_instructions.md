# Instrucciones de Compilación - Apple Silicon

## Instalación de Dependencias

### 1. Instalar Homebrew (si no está instalado)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Instalar OpenCV y CMake
```bash
brew install opencv cmake
```

### 3. Verificar instalación
```bash
pkg-config --modversion opencv4
cmake --version
```

## Compilación del Proyecto

### 1. Navegar al directorio
```bash
cd /ruta/a/croftondescriptor/apple_silicon_version
```

### 2. Configurar y compilar
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Solución de Problemas Comunes

### Error: OpenCV no encontrado
```bash
# Verificar instalación
brew list opencv

# Si no está instalado:
brew install opencv

# Configurar variables de entorno
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export OpenCV_DIR="/opt/homebrew/lib/cmake/opencv4"
```

### Error: Metal frameworks no encontrados
- Verificar que tienes macOS 11.0+
- Instalar Xcode Command Line Tools:
```bash
xcode-select --install
```

### Error: Compilador Objective-C++
```bash
# Verificar que tienes Xcode tools
xcode-select -p

# Reinstalar si es necesario
sudo xcode-select --reset
```

## Ejecución

### 1. Ejecutar (elige un target)
```bash
./build/crofton_simple tu_imagen.jpg simple_crofton_result.txt
./build/crofton_metal tu_imagen.jpg metal_crofton_result.txt
./build/crofton_optimized tu_imagen.jpg metal_optimized_result.txt
```

### 2. Verificar salida
- Se abrirán ventanas mostrando cada paso del procesamiento
- Se generará el archivo de salida indicado por parámetro

## Optimizaciones de Rendimiento

### Para Apple Silicon M1/M2/M3
El proyecto está optimizado automáticamente para Apple Silicon. CMake detecta la arquitectura y aplica las optimizaciones apropiadas:

- `-mcpu=apple-m1` para optimización específica
- `-O3` para máxima optimización
- `-march=native` para instrucciones específicas del procesador

### Verificar arquitectura
```bash
uname -m  # Debería mostrar 'arm64' en Apple Silicon
```

## Benchmarking

Para comparar rendimiento con la versión CUDA original:
```bash
time ./build/crofton_optimized tu_imagen.jpg metal_optimized_result.txt
```

El rendimiento esperado en Apple Silicon es significativamente mejor para operaciones de procesamiento de imágenes debido a la arquitectura unificada de memoria.

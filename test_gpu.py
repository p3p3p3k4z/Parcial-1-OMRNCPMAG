import os
import tensorflow as tf

# Ocultar avisos irrelevantes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("🔍 Buscando tarjetas gráficas disponibles para TensorFlow...")

# Obtener lista de GPUs físicas
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"\n✅ ¡ÉXITO! TensorFlow ha detectado {len(gpus)} tarjeta(s) gráfica(s) compatible(s) con CUDA:")
    for i, gpu in enumerate(gpus):
        print(f"   [{i}] -> {gpu.name}")
    print("\n🚀 El acelerador por hardware está LISTO para ser utilizado.")
else:
    print("\n❌ NO SE DETECTÓ NINGUNA GPU.")
    print("   TensorFlow seguirá ejecutándose en el procesador (CPU).")
    print("   Asegúrate de haber instalado 'tensorflow[and-cuda]' en tu entorno.")

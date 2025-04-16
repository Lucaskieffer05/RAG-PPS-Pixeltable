import pixeltable as pxt
import time
import sys
from typing import Optional

def hacer_pregunta(pregunta: str, timeout: int = 60) -> Optional[str]:
    """
    Versión corregida con:
    - Manejo correcto de condiciones NULL
    - Mejor seguimiento del estado
    - Robustez ante errores
    """
    try:
        sys.stdout.reconfigure(encoding='utf-8')
               
        queries_data = pxt.get_table('rag_demo.queries')
        
        # Insertar correctamente (forma recomendada)
        insert_result = queries_data.insert([{'question': pregunta}])
        row_id = insert_result[0]  # Obtenemos el ID real de la fila insertada
        print(f"\n⌛ Procesando pregunta (ID: {row_id})...", end="", flush=True)
        
        start_time = time.time()
        last_print = start_time
        
        while time.time() - start_time < timeout:
            # CONSULTA CORREGIDA (2 formas válidas):
            # Forma 1: Usando is_not_null() correctamente
            resultado = queries_data.where(queries_data._id == row_id) \
                                 .where(queries_data.response != None) \
                                 .select(queries_data.response) \
                                 .collect()
            
            # Forma 2: Más explícita
            # resultado = queries_data.where((queries_data._id == row_id) & 
            #                             (queries_data.response.is_not_null())) \
            #                      .select(queries_data.response) \
            #                      .collect()
            
            if resultado and resultado[0]['response']:
                print("\n")  # Salto de línea final
                return resultado[0]['response']
            
            if time.time() - last_print > 3:
                print(".", end="", flush=True)
                last_print = time.time()
            
            time.sleep(2)
        
        raise TimeoutError(f"Tiempo de espera agotado ({timeout}s)")
    
    except Exception as e:
        print(f"\n⚠️ Error: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        print("Sistema de Q&A con RAG")
        print("(Escribe 'salir' para terminar)\n")
        
        while True:
            pregunta = input("\nIngrese su pregunta: ").strip()
            
            if pregunta.lower() in ('salir', 'exit', 'quit'):
                print("\n¡Hasta luego!")
                break
                
            if not pregunta:
                print("Por favor ingrese una pregunta válida.")
                continue
                
            print(f"\n🔍 Buscando respuesta para: '{pregunta}'")
            
            respuesta = hacer_pregunta(pregunta, timeout=90)  # 90 segundos de timeout
            
            if respuesta:
                print("\n" + "═" * 80)
                print("💡 RESPUESTA:")
                print(respuesta)
                print("═" * 80)
            else:
                print("\nNo se pudo obtener una respuesta. Intente nuevamente.")
                
    except KeyboardInterrupt:
        print("\n\nPrograma terminado por el usuario.")
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
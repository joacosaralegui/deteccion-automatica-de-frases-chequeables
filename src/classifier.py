from nltk.tokenize import sent_tokenize
from math import ceil
import pickle
import os
from train_model import CustomLinguisticFeatureTransformer

MODEL_PATH = os.path.join('..','data','models', "Bernoulli NB.pkl")
#MODEL_PATH = os.path.join('..','data','models', "Logistic Regression.pkl")
# Load from file
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)
    print("**** Fact-checkable model loaded! *******")


def classify_text(text):
    sentences = sent_tokenize(text)
    sentences = [y for x in sentences for y in x.split("\n")]
    return classify_sentences(sentences)

    
def classify_sentences(sentences): 
    predictions = model.predict(sentences)
    
    labels = {
        'fact-checkable': True,
        'non-fact-checkable': False
    }

    classified = []
    for idx,c in enumerate(predictions):
        classified.append({
            'label': labels[c],
            'sentence': sentences[idx]
        })

    return classified

if __name__=="__main__":
    SENTENCES = [
        "El 99% de la gente que muere por un arma de fuego en la Argentina muere en manos de un delincuente que la asesina",
        "En el 2016 equiparamos el sueldo con la inflación; en el 2017, también",
        "En un ránking de las 50 ciudades más violentas, Brasil fue incluida en 17 ciudades. Nosotros no tenemos ni una sola ciudad incluida",
        "El 30% de las cárceles federales están llenas de extranjeros",
        "Argentina es el quinto país en formación de activos externos",
        "En la Argentina de hoy la palabra se ha devaluado peligrosamente.",
        "Parte de nuestra política se ha valido de la ella para ocultar la verdad o tergiversarla.",
        "Muchos creyeron que el discurso es una herramienta idónea para instalar en el imaginario público una realidad que no existe.",
        "Nunca midieron el daño que con la mentira le causaban al sistema democrático.",
        "Yo me resisto a seguir transitando esa lógica.",
        "Necesito que la palabra recupere el valor que alguna vez tuvo entre nosotros.",
        "Al fin y al cabo, en una democracia el valor de la palabra adquiere una relevancia singular.", 
        "Los ciudadanos votan atendiendo las conductas y los dichos de sus dirigentes.",
        "Toda simulación en los actos o en los dichos, representa una estafa al conjunto social que honestamente me repugna.",
        "He repetido una y otra vez que a mi juicio, en democracia, la mentira es la mayor perversión en la que puede caer la política.",
        "Gobernar no es mentir ni es ocultarle la verdad al pueblo.",
        "Gobernar es admitir la realidad y transmitirla tal cual es para poder transformarla en favor de una sociedad que se desarolle en condiciones de mayor igualdad."
    ]

    text = """
    extensa de lo que inicialmente se pensó. Ni un solo día bajamos los brazos. Ni ante la inclemencia del contagio, ni ante la crítica injusta.
    Sin aislamiento y distanciamiento hubiera habido mayor la velocidad en los contagios y un sistema de salud que estaba en condiciones de abandono hubiera colapsado. Cuando los sistemas colapsan, la mortalidad aumenta de manera significativa.
    Este no es el logro de un Gobierno sino el de una Nación puesta de pie para superar adversidades.
    Trabajamos en cada decisión con las gobernadoras y los gobernadores de las 24 jurisdicciones, a quienes aprovecho esta oportunidad para brindarles mi más sincero reconocimiento.
    Incorporamos más de 4.000 unidades de terapia intensiva, lo que implicó un aumento del 47 % en la capacidad instalada. Construimos 12 hospitales modulares en tiempo récord.
    Nuestros trabajadores y trabajadoras de la salud dieron un ejemplo en la frontera más expuesta de la pandemia.
    Nuestros empresarios se movilizaron para brindar asistencia alimentaria de emergencia junto al Estado, organizaciones gremiales, iglesias y movimientos populares, y para impulsar la producción argentina de 3.300 respiradores.
    Nuestros científicos se unieron en redes de investigación para producir kits de detección temprana, barbijos, tratamientos de la enfermedad como es el caso del suero equino hiperinmune, e innumerables aportes de todas las disciplinas.
    Nuestras Fuerzas Armadas protagonizaron el operativo militar más importante desde la gesta de Malvinas, para acercar apoyo logístico, humanitario y social en los barrios más populares.
    Las fuerzas de seguridad federales trabajaron articuladamente con todas las provincias y jurisdicciones, con niveles de exposición muy elevados.
    Nuestros diplomáticos cooperaron para repatriar a 205 mil personas en los primeros meses de la pandemia, en el operativo de asistencia consular más grande de la historia argentina.
    Las universidades hicieron una veloz transición hacia la enseñanza virtual, organizaron el voluntariado en diferentes zonas del país, contribuyeron a procesar tests y, articulando con el CONICET, produjeron contribuciones científicas muy relevantes.
    Ante la necesidad de suspender las clases presenciales –situación que también se verificó en 190 países- el gobierno nacional y las 24 jurisdicciones desplegaron recursos educativos en soporte digital, papel, televisivo y radial, para estudiantes, familias y docentes.
    Se implementaron medidas para acompañar a quienes contaban con tecnología y conectividad, así como a quienes, en contextos de vulnerabilidad o aislamiento geográfico, necesitaban otras opciones.
    Así, el compromiso de los equipos docentes y directivos, el esfuerzo de estudiantes y la dedicación de las familias, fue inmenso.
    Sé muy bien que puede resultar difícil valorar aquello que no sucedió. Es difícil porque las consecuencias más graves que evitamos obviamente no se ven. Salvo que hagamos memoria y comparemos nuestra experiencia con las imágenes tenebrosas que llegaron desde otros países.
    Para todas estas argentinas y estos argentinos que han desplegado su corazón al servicio de los demás, les pido por favor que brindemos un sentido aplauso, para que se sienta nuestro reconocimiento a lo largo y ancho del país.
    En este tiempo, personalmente, he sufrido con cada fallecimiento. Para mí quienes perdieron la vida en la pandemia nunca fueron números o estadísticas. Siempre fueron seres humanos, con historias personales y afectos. Otra vez, manifiesto aquí mi acompañamiento para quienes han perdido un ser querido.
    Mientras el trabajo común nos convocaba a millones, debimos enfrentar a esos mismos sectores que pretendieron desmoralizar al ciudadano medio hablando también de la ausencia de una estrategia económica.
    Aquel reproche fue y es definitivamente inmerecido. Junto al cúmulo de medidas sanitarias, fuimos también capaces de impulsar medidas económicas y de protección social para paliar los efectos de la inédita calamidad que atravesábamos.
    Dispusimos en marzo el congelamiento de los precios de alimentos, bebidas, productos de limpieza e higiene personal.
    Decretamos la creación del Ingreso Familiar de Emergencia que consistió en tres pagos de $10.000 que alcanzó a más de 9 millones de personas. Nunca en la historia argentina se había creado una política de ese alcance.
    Suspendimos los cortes de los servicios públicos por falta de pago en los segmentos vulnerables de la población.
    Prohibimos los despidos sin causa o por fuerza mayor. La doble indemnización ya regía desde el 13 de diciembre de 2019.
    Creamos el programa de Asistencia de Emergencia al Trabajo y la Producción. El 70 % de quienes son empleadores en Argentina recibieron asistencia del Estado a través del pago del salario complementario a sus trabajadoras y trabajadores, préstamos subsidiados y/o reducción o postergación de las cargas patronales. El 99,5 % de las empresas asistidas fueron PyMES.
    En el marco del ATP también creamos una línea de crédito a tasa cero por hasta $150.000 para las y los trabajadores independientes formales (monotributistas y autónomos). A las y los trabajadores del sector cultura, les dimos un plazo de gracia de 12 meses.
    Para atender la demanda alimentaria de la población más vulnerable multiplicamos el suministro de alimentos en la red de comedores comunitarios y escolares, en articulación con las provincias, los municipios y las organizaciones sociales. Se dieron refuerzos de la Tarjeta Alimentar, la AUH y la jubilación mínima. Creció más del 400 % el crédito inicial programado destinado a políticas alimentarias.
    Además, con el programa ProHuerta apoyamos a 610 mil huertas familiares, escolares y comunitarias.
    """

    print("Started classification...")
    classified = classify_sentences(SENTENCES)
    print("Classification ended succesfully!")
    print("Classifiyng text")
    classified = classify_text(text)
    print("END!")
    import pdb; pdb.set_trace()
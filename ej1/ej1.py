# vector datos
data = []
# vector columnas
column=[]
#arch=open("heart.csv","r")
arch=open("iris.csv","r")

sw=0
for l in arch:
    l=l.strip()
    #print(l,sw)
    if(sw==0):
        sw=1
        column=list(l.split(";"))
    else:
        #lista=list(map(float,l.split(",")))
        lista=list(l.split(";"))
        data.append(lista)
        
# n = nro de filas - restando la primera fila de cabecera
n = (len(data))
# m = nro de columnas
m = (len(data[0]))
#----------------------
from sklearn import datasets

datos = datasets.load_iris()
datosX=datos.data
datosY=datos.target

#----------------------
"""
import numpy as np

def sigmoide(x):
    return 1/(1+np.exp(-x))

#Parametros
n_hidden = 2 #nro u. en la capa escondida
epocas= 1000 
taza_aprend = 0.05 #taza de aprendizaje

ult_costo=None

m,k = datosX.shape
#Inicializacion de los pesos
entrada_escondida=np.random.normal(scale=1/k**0.5,size=(k,n_hidden))
#bias
escondida_salida=np.random.normal(scale=1/k**0.5,size=(k,n_hidden))

#Entrenamiento
for e in range(epocas):
    #Variables para el gradiente
    gradiente_entrada_escondida=np.zeros(entrada_escondida.shape)
    gradiente_escondida_salida=np.zeros(escondida_salida.shape)
    
    #Itera sobre el conjunto de entrenamiento
    for x,y in zip(datosX,datosY):
        #Pasada hacia adelante (forward pass)
        z=sigmoide(np.matmul(x,entrada_escondida))
        y_p=sigmoide(np.matmul(escondida_salida,z)) #prediccion
        #Pasada hacia atras      
        salida_error=(y-y_p)*y_p*(1-y_p)
        
        escondida_error=np.dot(salida_error, escondida_salida)*z*(1-z)
        
        gradiente_entrada_escondida+=escondida_error*x[:,None]
        gradiente_escondida_salida+=salida_error*z
    #Actualiza los parametros(pesos)
    entrada_escondida+=taza_aprend*gradiente_entrada_escondida/m
    escondida_salida+=taza_aprend*gradiente_escondida_salida/m
    
    if(e%(epocas/10)==0):
        z=sigmoide(np.dot(datosX.values, entrada_escondida))
        y_p=sigmoide(np.dot(z,escondida_salida))
        
        #Funcion Costo
        costo=np.mean((y_p-datosY)**2)
        if(ult_costo and ult_costo < costo):
            print("Costo de Entrenamiento: ",costo," Advertencia")
        else:
            print("Costo de Entrenamiento: ",costo)
        ult_costo=costo
        
    #Precision en lso datos de prueba
    z=sigmoide(np.dot(datosX,entrada_escondida))
    y_p=sigmoide(z,escondida_salida)
    
    predicciones=y_p>0.5
    precision=np.mean(predicciones==datosY)
    print(f"Precision: {precision:.3f}")
"""
#-----------------
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_perdida(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()

class redNeuronal:

  def __init__(self):
    # Pesos
    self.p1 = np.random.normal()
    self.p2 = np.random.normal()
    self.p3 = np.random.normal()
    self.p4 = np.random.normal()
    self.p5 = np.random.normal()
    self.p6 = np.random.normal()
    
    self.p7 = np.random.normal()
    self.p8 = np.random.normal()
    self.p9 = np.random.normal()
    self.p10 = np.random.normal()
    
    # Bias
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def retroalimentacion(self, x):
    neurona1 = sigmoid(self.p1 * x[0] + self.p2 * x[1] + self.p3 * x[2] + self.p4 * x[3] + self.b1)
    neurona2 = sigmoid(self.p5 * x[0] + self.p6 * x[1] + self.p7 * x[2] + self.p8 * x[3]+  self.b2)
    neurona3 = sigmoid(self.p9 * neurona1 + self.p10 * neurona2 + self.b3)
    return neurona3

  def train(self, datos, y_trues):
    taza_aprendizaje = 0.4
    epocas = 1000 

    for epoca in range(epocas):
	
      for x, y_true in zip(data, y_trues):
        sum_neurona1 = self.p1 * x[0] + self.p2 * x[1] + self.p3 * x[2] + self.p4 * x[3] + self.b1
        neurona1 = sigmoid(sum_neurona1)

        sum_neurona2 = self.p5 * x[0] + self.p6 * x[1] + self.p7 * x[2] + self.p8 * x[3]+  self.b2
        neurona2 = sigmoid(sum_neurona2)

        sum_neurona3 = self.p9 * neurona1 + self.p10 * neurona2 + self.b3
        neurona3 = sigmoid(sum_neurona3)
        y_pred = neurona3

        # derivada parcial
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neurona3
        d_ypred_d_p9 = neurona1 * deriv_sigmoid(sum_neurona1)
        d_ypred_d_p10 = neurona2 * deriv_sigmoid(sum_neurona1)
        d_ypred_d_b3 = deriv_sigmoid(sum_neurona3)

        d_ypred_d_neurona1 = self.p9 * deriv_sigmoid(sum_neurona3)
        d_ypred_d_neurona2 = self.p10 * deriv_sigmoid(sum_neurona3)

        # Neurona1
        d_neurona1_d_p1 = x[0] * deriv_sigmoid(sum_neurona1)
        d_neurona1_d_p2 = x[1] * deriv_sigmoid(sum_neurona1)
        d_neurona1_d_p3 = x[2] * deriv_sigmoid(sum_neurona1)
        d_neurona1_d_p4 = x[3] * deriv_sigmoid(sum_neurona1)
        d_neurona1_d_b1 = deriv_sigmoid(sum_neurona1)

        # Neurona2
        d_neurona2_d_p5 = x[0] * deriv_sigmoid(sum_neurona2)
        d_neurona2_d_p6 = x[1] * deriv_sigmoid(sum_neurona2)
        d_neurona2_d_p7 = x[2] * deriv_sigmoid(sum_neurona2)
        d_neurona2_d_p8 = x[3] * deriv_sigmoid(sum_neurona2)
        d_neurona2_d_b2 = deriv_sigmoid(sum_neurona2)

        # Actualizar
        # Neurona1
        self.p1 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p1
        self.p2 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p2
        self.p3 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p3
        self.p4 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p4
        self.b1 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_b1

        # Neurona2
        self.p5 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p5
        self.p6 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p6
        self.p7 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p7
        self.p8 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p8
        self.b2 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_b2

        # Neurona3
        self.p9 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_p9
        self.p10 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_p10
        self.b3 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_b3

      # perdida por cada epoca
      if epoca % 10 == 0:
        y_preds = np.apply_along_axis(self.retroalimentacion, 1, data)
        perdida = mse_perdida(y_trues, y_preds)
        print("epoca %d perdida: %.3f" % (epoca, perdida))
        
#------
from sklearn import datasets

datos = datasets.load_iris()
datosX=datos.data
datosY=datos.target

#preparar datos
from sklearn.model_selection import train_test_split
entrenarX, testX, entrenarY, testY = train_test_split(datosX, datosY, test_size=0.3)


# Entrenar
data = datosX
y_trues = datosY

red = redNeuronal()
red.train(datosX, datosY)
#red.train(entrenarX, entrenarY)

# Predecir
#data1 = np.array([4.0,3.5,1.4,0.7])
data1 = testX
print(data1)
for i in data1:
    print(f"Data {i}: {red.retroalimentacion(i):.3f}" )

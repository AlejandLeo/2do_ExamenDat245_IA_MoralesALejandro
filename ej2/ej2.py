# vector datos
data = []
# vector columnas
column=[]
#arch=open("heart.csv","r")
arch=open("The_Cancer_data_1500_V2.csv","r")

sw=0
for l in arch:
    l=l.strip()
    #print(l,sw)
    if(sw==0):
        sw=1
        column=list(l.split(","))
    else:
        lista=list(map(float,l.split(",")))
        #lista=list(l.split(","))
        data.append(lista)
# n = nro de filas - restando la primera fila de cabecera
n = (len(data))
# m = nro de columnas
m = (len(data[0]))
#----------------------
datosX=[]
datosY=[]
for i in range(n):
    datosX.append(data[i][:m-1])
    datosY.append(data[i][m-1])
#-----------------
import numpy as np
#"""
def faescalon(x):
    return 1 if x >= 0 else 0

def deriv_faescalon(x):
    return 0  
"""
def faescalon(x):
  return 1 / (1 + np.exp(-x))

def deriv_faescalon(x):
  fx = faescalon(x)
  return fx * (1 - fx)
"""
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
    
    self.p10 = np.random.normal()
    self.p11 = np.random.normal()
    self.p12 = np.random.normal()
    self.p13 = np.random.normal()
    self.p14 = np.random.normal()
    self.p15 = np.random.normal()
    self.p16 = np.random.normal()
    self.p17 = np.random.normal()
    self.p18 = np.random.normal()
    # Bias
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def retroalimentacion(self, x):
    neurona1 = faescalon(self.p1 * x[0] + self.p2 * x[1] + self.p3 * x[2] + self.p4 * x[3] + self.p5 * x[4] + self.p6 * x[5] + self.p7 * x[6] + self.p8 * x[7] + self.b1)
    neurona2 = faescalon(self.p9 * x[0] + self.p10 * x[1] + self.p11 * x[2] + self.p12 * x[3] + self.p13 * x[4] + self.p14 * x[5] + self.p15 * x[6] + self.p16 * x[7] + self.b2)
    neurona3 = faescalon(self.p17 * neurona1 + self.p18 * neurona2 + self.b3)
    return neurona3

  def train(self, datos, y_trues):
    taza_aprendizaje = 0.2
    epocas = 1000 

    for epoca in range(epocas):
	
      for x, y_true in zip(data, y_trues):
        sum_neurona1 = self.p1 * x[0] + self.p2 * x[1] + self.p3 * x[2] + self.p4 * x[3] + self.p5 * x[4] + self.p6 * x[5] + self.p7 * x[6] + self.p8 * x[7] + self.b1
        neurona1 = faescalon(sum_neurona1)

        sum_neurona2 = self.p9 * x[0] + self.p10 * x[1] + self.p11 * x[2] + self.p12 * x[3] + self.p13 * x[4] + self.p14 * x[5] + self.p15 * x[6] + self.p16 * x[7] + self.b2
        neurona2 = faescalon(sum_neurona2)

        sum_neurona3 = self.p17 * neurona1 + self.p18 * neurona2 + self.b3
        neurona3 = faescalon(sum_neurona3)
        y_pred = neurona3

        # derivada parcial
        d_L_d_ypred = -2 * (y_true - y_pred)

        #actaulizacion de datos
        
        # Neurona3
        d_ypred_d_p17 = neurona1 * deriv_faescalon(sum_neurona1)
        d_ypred_d_p18 = neurona2 * deriv_faescalon(sum_neurona1)
        d_ypred_d_b3 = deriv_faescalon(sum_neurona3)

        d_ypred_d_neurona1 = self.p17 * deriv_faescalon(sum_neurona3)
        d_ypred_d_neurona2 = self.p18 * deriv_faescalon(sum_neurona3)

        # Neurona1
        d_neurona1_d_p1 = x[0] * deriv_faescalon(sum_neurona1)
        d_neurona1_d_p2 = x[1] * deriv_faescalon(sum_neurona1)
        d_neurona1_d_p3 = x[2] * deriv_faescalon(sum_neurona1)
        d_neurona1_d_p4 = x[3] * deriv_faescalon(sum_neurona1)
        d_neurona1_d_p5 = x[4] * deriv_faescalon(sum_neurona1)
        d_neurona1_d_p6 = x[5] * deriv_faescalon(sum_neurona1)
        d_neurona1_d_p7 = x[6] * deriv_faescalon(sum_neurona1)
        d_neurona1_d_p8 = x[7] * deriv_faescalon(sum_neurona1)
        d_neurona1_d_b1 = deriv_faescalon(sum_neurona1)

        # Neurona2
        d_neurona2_d_p9 = x[0] * deriv_faescalon(sum_neurona2)
        d_neurona2_d_p10 = x[1] * deriv_faescalon(sum_neurona2)
        d_neurona2_d_p11 = x[2] * deriv_faescalon(sum_neurona2)
        d_neurona2_d_p12 = x[3] * deriv_faescalon(sum_neurona2)
        d_neurona2_d_p13 = x[4] * deriv_faescalon(sum_neurona2)
        d_neurona2_d_p14 = x[5] * deriv_faescalon(sum_neurona2)
        d_neurona2_d_p15 = x[6] * deriv_faescalon(sum_neurona2)
        d_neurona2_d_p16 = x[7] * deriv_faescalon(sum_neurona2)
        d_neurona2_d_b2 = deriv_faescalon(sum_neurona2)

        # Actualizar
        # Neurona1
        self.p1 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p1
        self.p2 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p2
        self.p3 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p3
        self.p4 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p4
        self.p5 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p5
        self.p6 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p6
        self.p7 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p7
        self.p8 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p8
        self.b1 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_b1

        # Neurona2
        self.p9 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p9
        self.p10 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p10
        self.p11 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p11
        self.p12 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p12
        self.p13 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p13
        self.p14 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p14
        self.p15 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p15
        self.p16 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p16
        self.b2 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_b2

        # Neurona3
        self.p17 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_p17
        self.p18 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_p18
        self.b3 -= taza_aprendizaje * d_L_d_ypred * d_ypred_d_b3

      # perdida por cada epoca
      if epoca % 10 == 0:
        y_preds = np.apply_along_axis(self.retroalimentacion, 1, data)
        perdida = mse_perdida(y_trues, y_preds)
        print("epoca %d perdida: %.3f" % (epoca, perdida))
        
#------
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
for i in data1:
    print(f"Data {i}: {red.retroalimentacion(i):.3f}" )

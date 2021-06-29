import tkinter as tk
from PIL import ImageTk, Image, ImageDraw
import PIL
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from scipy import ndimage
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

def crop(data):
    data = data.reshape(28,28)
    r = data[~np.all(data == 0, axis=1)] # loại bỏ hàng toàn 0
    idx = np.argwhere(np.all(r[..., :] == 0, axis=0)) # vị trí cột toàn 0
    c = np.delete(r, idx, axis=1) # loại bỏ cột toàn 0
    res = cv2.resize(c, dsize=(28, 28)) # Trả lại về size 28x28
    res = res.flatten() # Chuyển lại thành array 1 chiều như ban đầu
    return res

def cropData(data):
    return np.apply_along_axis(crop, 1, data)

def Hog(data):
    img = data.reshape(28,28,1)
    resized_img = resize(img, (24*4, 12*4))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd

def HogData(data):
    return np.apply_along_axis(Hog, 1, data)

N = 28*10

class draw_predict(tk.Tk):
	pil_image = PIL.Image.new("L", (N, N))
	pil_draw = ImageDraw.Draw(pil_image)
	labelText = "Press and draw"

	def __init__(self):
		tk.Tk.__init__(self)
		self.canvas = tk.Canvas(self, width=N, height=N, bg = "black")
		self.canvas.pack()
		
		self.canvas.pack(side="top", fill="both", expand=True)
		self.canvas.bind("<B1-Motion>", self.draw)

		self.button=tk.Button(text="save",command=self.save)
		self.button.pack()

		self.button_predict = tk.Button(self, text = "Guess it!", command = self.make_prediction)
		self.button_predict.pack( fill="both", expand=True)

		self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
		self.button_clear.pack( fill="both", expand=True)

		self.message = tk.Label(self,text = self.labelText)
		self.message.pack(side = 'bottom')

	def draw(self,event):
		color = 'white'
		x1,y1 = (event.x - 2), (event.y - 2)
		x2,y2 = (event.x + 2), (event.y + 2)
		self.canvas.create_rectangle(x1,y1,x2,y2,fill = color, outline = color)
		self.pil_draw.rectangle([(x1,y1),(x2,y2)],fill= color)
	
	def save(self):
		print(np.asarray(self.pil_image.resize((28,28))))
		self.canvas.update()
		filename = "image.png"
		self.pil_image.resize((28,28)).save(filename)

	def clear_all(self):
		self.canvas.delete("all")
		self.pil_image = PIL.Image.new("L", (N, N))
		self.pil_draw = ImageDraw.Draw(self.pil_image)

	def make_prediction(self):
		a = np.asarray(self.pil_image.resize((28,28))).reshape(1,-1)
		filename = 'preprcess.sav'
		preprocess = joblib.load(filename)
		a = preprocess.transform(a.astype('float64'))

		filename = 'hog_model.sav'
		loaded_model = joblib.load(filename)
		print(a)
		predict = loaded_model.predict(a)
		self.message['text'] = "It looks like number " + predict[0]


a = draw_predict()
a.mainloop()
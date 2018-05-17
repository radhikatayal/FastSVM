import os
from tkinter import *
from tkinter import ttk
from FSVMDiabetes import DiabetesPrediction
from FSVMBCancer import BCancerPrediction
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from PIL import ImageTk, Image
from tkinter import filedialog

class App:

    def __init__(self, master):
        master.title("Diabetes Tool")
        #self.D = DiabetesPrediction()
        self.D = BCancerPrediction()
        self.acc, self.sens, self.spec, self.roc_auc = self.D.process()
        X1, ytrue, self.sc_X = self.D.data_processing()
        #top = master.winfo_toplevel()
        #top.rowconfigure(0, weight=1)
        #top.columnconfigure(0, weight=1)
        #master.rowconfigure(0, weight=1)
        #master.columnconfigure(0, weight=1)

        left = Frame(master)
        left.grid(column=0, row=0, sticky=(N, W, E, S), columnspan=1, padx=10)
        right = Frame(master)
        right.grid(column=1, row=0, sticky=(N, W, E, S), columnspan=1, padx=10)

        lbl = Label(right, text="Diabetes Detection tool", font=("Arial Bold", 18))
        lbl.grid(row=0, sticky=S+E+W, columnspan=2, pady = 20)
        lbl = Label(right, text="", font=("Arial Bold", 10))
        lbl.grid(row=1, sticky=S+E+W, columnspan=2, pady = 40)

        self.font = font=("Arial", 14)
        label_preg = Label(right, text="Number of times pregnant: ", relief=RIDGE, font=self.font)#.pack(side=LEFT)
        label_preg.grid(column=0, row=2, sticky=W, padx = 20, pady = 10)
        self.preg = IntVar()
        input1 = Entry(right, textvariable=self.preg, relief=RIDGE, font = self.font)
        input1.grid(column=1, row=2, sticky=E, padx = 20, pady = 10)
        input1.focus_set()
        #self.preg.set(6)

        Label(right, text="Plasma glucose concentration: ", relief=RIDGE, font = self.font).grid(column=0, row=3, sticky=W, padx = 20, pady = 10)#.pack(side=LEFT)
        self.pgc = DoubleVar()
        input2 = Entry(right, textvariable=self.pgc, relief=RIDGE, font = self.font)
        #self.input.pack(side=LEFT)
        input2.grid(column=1, row=3, sticky=E, padx = 20, pady = 10)
        #self.pgc.set(148)

        Label(right, text="Diastolic blood pressure: ", relief=RIDGE, font = self.font).grid(column=0, row=4, sticky=W, padx = 20, pady = 10)#.pack(side=LEFT)
        self.bp = DoubleVar()
        input3 = Entry(right, textvariable=self.bp, relief=RIDGE, font = self.font)
        #self.input.pack(side=LEFT)
        input3.grid(column=1, row=4, sticky=E, padx = 20, pady = 10)
        #self.bp.set(72)

        Label(right, text="Body mass index: ", relief=RIDGE, font = self.font).grid(column=0, row=5, sticky=W, padx = 20, pady = 10)#.pack(side=LEFT)
        self.bmi = DoubleVar()
        input4 = Entry(right, textvariable=self.bmi, relief=RIDGE, font = self.font)
        #self.input.pack(side=LEFT)
        input4.grid(column=1, row=5, sticky=E, padx = 20, pady = 10)
        #self.bmi.set(33.5)

        Label(right, text="Diabetes pedigree function: ", relief=RIDGE, font = self.font).grid(column=0, row=6, sticky=W, padx = 20, pady = 10)#.pack(side=LEFT)
        self.pedigree = DoubleVar()
        input5 = Entry(right, textvariable=self.pedigree, relief=RIDGE, font = self.font)
        #self.input.pack(side=LEFT)
        input5.grid(column=1, row=6, sticky=E, padx = 20, pady = 10)
        #self.pedigree.set(.627)

        Label(right, text="Patient Age: ", relief=RIDGE, font = self.font).grid(column=0, row=7, sticky=W, padx = 20, pady = 10)#.pack(side=LEFT)
        self.age = IntVar()
        input6 = Entry(right, textvariable=self.age, relief=RIDGE, font = self.font)
        #self.input.pack(side=LEFT)
        input6.grid(column=1, row=7, sticky=E, padx = 20, pady = 10)
        #self.age.set(50)

        self.button = Button(
            right, text="QUIT", fg="red", command=master.quit, font = self.font
            )
        #self.button.pack()
        self.button.grid(column=0, row=8, pady = 20)

        self.predict_btn = Button(right, text="Predict Diabetes", command=self.predict, font = self.font)
        self.predict_btn.grid(column=1, row=8, pady = 20)

        Label(right, text="Patient Diabetes: ", font = self.font).grid(column=0, row=9, sticky=E, pady = 10)#.pack(side=LEFT)
        self.diabetes = StringVar()
        self.diabetes.set("-")
        Label(right, textvariable=self.diabetes, font = self.font).grid(column=1, row=9, sticky=W, pady=10)

        Label(right, text="Confidence level: ", font = self.font).grid(column=0, row=10, sticky=E, pady = 10)#.pack(side=LEFT)
        self.diabetes_proba = StringVar()
        self.diabetes_proba.set("--%")
        Label(right, textvariable=self.diabetes_proba, font = self.font).grid(column=1, row=10, sticky=W, pady=10)

        #self.original = Image.open("semi_supervised_boundary_plot.png")
        #basewidth = 400
        #wpercent = (basewidth / float(self.original.size[0]))
        #hsize = int((float(self.original.size[1]) * float(wpercent)))
        #self.original = self.original.resize((basewidth, hsize), Image.ANTIALIAS)
        #self.img = ImageTk.PhotoImage(self.original, width=self.original.size[0], height=self.original.size[1])
        #Canvas()
        #self.display = Canvas(left, bd=0, highlightthickness=0, width=self.original.size[0]+100,height=self.original.size[1])
        #self.display.create_image(100, 0, image=self.img, anchor=NW, tags="IMG")
        #self.display.grid(column=0, row=0, sticky=W + E + N + S)
        #self.display.image = self.img
        #self.display.place(width=self.original.size[0]/2,height=self.original.size[1]/2)
        #master.bind("<Configure>", self.resize)
        #panel = Label(master, image=img)
        #panel.image = img

        #panel.pack(side="bottom", fill="both", expand="yes")
        #panel.grid(column=0, row=7, sticky=N+S+E+W)
        #Widget(master, img)
        Label(left, text="Diabetes Prediction Model", font=("Arial Bold", 18)).grid(row=0, sticky=S+E+W, columnspan=3, pady = 20)
        Label(left, text="Load Model Data: ", relief=RIDGE, font = self.font).grid(column=0, row=1, sticky=E, padx = 20, pady = 10)#.pack(side=LEFT)

        Button(left, text="Browse", command=self.browsefunc, font = self.font).grid(column=1, row=1, sticky=W, padx = 20, pady = 10)
        Checkbutton(left, text = "Annomization",font = self.font).grid(column=2, row=1, sticky=W, padx = 20, pady = 10)
        self.filename = StringVar()
        #X, ytrue, sc_X = self.data_processing(self.filename)
        self.filename.set("")# + X.shape())
        self.pathlabel = Label(left, textvariable= self.filename, font = self.font)
        self.pathlabel.grid(column=0, row=2, padx=20, pady=10, columnspan=2)
        self.left = left

    def browsefunc(self):
        filepath = filedialog.askopenfilename(initialdir="/", title="Select file to load")
        filename = os.path.basename(filepath)
        if filename.lower().endswith(('.csv')):
            self.filename.set("Data Loaded Successfully, filename: " + filename)
            fig = Figure(figsize=(6,6))
            a = fig.add_subplot(111)
            a = self.D.plot_boundary(a)
            canvas = FigureCanvasTkAgg(fig, master=self.left)
            #canvas.grid(row=1, sticky=S+E+W, columnspan=2, pady = 20)
            canvas.get_tk_widget().grid(column=0, row=3, sticky=W + E + N + S, columnspan=3)
            canvas.draw()

            Label(self.left, text="Model Accuracy: ", font = self.font).grid(column=0, row=4, sticky=E, pady = 10)#.pack(side=LEFT)
            self.accuracy = StringVar()
            self.accuracy.set(str(self.acc*100)+"%")
            Label(self.left, textvariable=self.accuracy, font = self.font).grid(column=1, row=4, sticky=W, pady=10)
            Label(self.left, text="ROC AUC percentage: ", font = self.font).grid(column=0, row=5, sticky=E, pady = 10)#.pack(side=LEFT)
            self.roc_auc_p = StringVar()
            self.roc_auc_p.set(str(self.roc_auc*100)+"%")
            Label(self.left, textvariable=self.roc_auc_p, font = self.font).grid(column=1, row=5, sticky=W, pady=10)
            Label(self.left, text="Sensitivity: ", font = self.font).grid(column=0, row=6, sticky=E, pady = 10)#.pack(side=LEFT)
            self.sensitivity = StringVar()
            self.sensitivity.set(str(self.sens*100)+"%")
            Label(self.left, textvariable=self.sensitivity, font = self.font).grid(column=1, row=6, sticky=W, pady=10)
        else:
            self.filename.set("Data Loading UnSuccessfull, filename: " + filename + ",Try Again")

        #self.pathlabel.config(text=self.filename)

    def predict(self):
        sample = [[self.preg.get(), self.pgc.get(), self.bp.get(), self.bmi.get(), self.pedigree.get(), self.age.get()]]
        sample = self.sc_X.transform(sample)
        y = self.D.predict(sample)
        y_proba = self.D.ssmodel.predict_proba(sample)*100
        if (y == 0):
            self.diabetes.set("NO")
        else:
            self.diabetes.set("YES")
        s = str(y_proba[0][1]) + "%"
        self.diabetes_proba.set(s)
        print("Diabetes prediction for following values:",
              self.preg.get(), self.pgc.get(), self.bp.get(), self.bmi.get(), self.pedigree.get(), self.age.get())

    def resize(self, event):
        size = (event.width, event.height)
        resized = self.original.resize(size,Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(resized)
        self.display.delete("IMG")
        self.display.create_image(0, 0, image=self.img, anchor=NW, tags="IMG")


    #def quit(event):
    #    if tkMessageBox.askokcancel('Quit','Do you really want to quit?'):
    #        root.destroy()
root = Tk()
#root.geometry('2000x3000')
app = App(root)

root.mainloop()

root.destroy() # optional; see description below
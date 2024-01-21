import tkinter as tk

class MyGUI:
    def __init__(self):
            self.MyEmail="something"
            self.root= tk.Tk()
            self.root.geometry("500x500")
            self.root.title("Face_recognition_system")
            self.label =tk.Label(self.root,text="Enter your email",font=('Arial',18))
            self.label.place(x=0,y=0,height=100,width=180)
            self.myentry=tk.Entry(self.root)
            self.myentry.pack(padx=20,pady=40)
            self.button=tk.Button(self.root,text="Enter",font=('Arial',18), command=self.show_message)
            self.button.pack(padx=10,pady=30)
         

            self.root.mainloop()


    def show_message(self):
                self.MyEmail=self.myentry.get()
                print(self.MyEmail)
                self.root.destroy()

    def return_email(self):
                return self.MyEmail


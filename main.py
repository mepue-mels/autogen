from include import *
import tkinter as tk

url = decrypt_url()

def main():
	if connectivity_test(url):
		root = tk.Tk()
		root.geometry("1280x720")

		label = tk.Label(root,
				 text="Automatic Question Generation",
				 font=("Helvetica", 64),
				 )

		label.pack(pady=150)

		button = tk.Button(root, text="Start", font=("Helvetica", 24), width=10, height=2)
		button.pack()

		root.mainloop()
	else:
		print("Error: cloud not active")

#entry point for window
if __name__ == "__main__":
	main()


from include import *
import tkinter as tk

"""
main.py

Serves as the front end (Tkinter) for the app that communicates with the GCP Cloud Run service for the LLM service.

Entry point: main()
"""

url = decrypt_url()

def show_frame(frame):
	frame.tkraise()

def main():
	if connectivity_test(url): #entry point

		#main canvas
		root = tk.Tk()
		root.geometry("1280x720")


		#define the frames here
		frame_entry = tk.Frame(root)
		frame_camera = tk.Frame(root)


		#main frame elements
		MAIN_label = tk.Label(frame_entry,
				 text="Automatic Question Generation",
				 font=("Helvetica", 64),
				 )

		MAIN_label.pack(pady=150)

		MAIN_button = tk.Button(frame_entry,
					text="Start",
					font=("Helvetica", 24),
					width=10,
					height=2,
					command=lambda: show_frame(frame_camera))
		MAIN_button.pack()

		#elements for camera frame


		#stack the elements from the grid
		for frame in (frame_entry, frame_camera):
			frame.grid(row=0, column=0, sticky="nsew")


		#unrestrict the main canvas for resizing using weights
		root.rowconfigure(0, weight=1)
		root.columnconfigure(0, weight=1)

		show_frame(frame_entry)

		root.mainloop()
	else:
		print("Error: cloud not active")

#entry point for window
if __name__ == "__main__":
	main()


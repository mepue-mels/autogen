from include import *

url = decrypt_url()


def main():
	if connectivity_test(url):
		print("yes")
	else:
		print("no")


#entry point for window
if __name__ == "__main__":
	main()


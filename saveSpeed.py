from sys import argv, exit


if __name__ == '__main__':
	fileName = argv[1]

	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	for line in A:
		print(float(line))
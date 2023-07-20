bullshit := cfitsio-4.2.0/*.o -Icfitsio-4.2.0 -lm -lz -lOpenCL
fitstest_linux: fitstest.c fitscl.c
	gcc -g fitstest.c fitscl.c timing.c $(bullshit) -o fitstest_linux
imstat:
	gcc imstat.c $(bullshit) -o imstat

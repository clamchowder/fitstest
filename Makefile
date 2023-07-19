bullshit := cfitsio-4.2.0/*.o -Icfitsio-4.2.0 -lm -lz -lOpenCL
fitstest: fitstest.c
	gcc -g fitstest.c fitscl.c $(bullshit) -o fitstest
imstat:
	gcc imstat.c $(bullshit) -o imstat

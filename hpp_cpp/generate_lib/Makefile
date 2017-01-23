HIP_PATH?=$(shell hipconfig --path)

HIPCC= $(HIP_PATH)/bin/hipcc


all: libkernel.a


#hipify foo.cu > foo.cpp
# Then review & finish port in foo.cpp

foo.o: foo.cpp
	$(HIPCC) -c foo.cpp -o $@

goo.o: goo.cpp
	$(HIPCC) -c goo.cpp -o $@

libkernel.a:foo.o goo.o
	ar rcs $@ $^

clean:
	rm -f *.o *.out *~	

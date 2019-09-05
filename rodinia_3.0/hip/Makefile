#space-separated list of tests to run:
USECOLORS?=1
ifeq ($(USECOLORS),1)
COLOR=\033[0;35m
NOCOLOR=\033[0m
RED=\033[0;31m
GREEN=\033[0;32m
endif
ifeq ($(HIP_PLATFORM),nvcc)
BLACKLIST=hybridsort leukocyte 
else
BLACKLIST=leukocyte particlefilter
endif
TESTS=$(filter-out $(BLACKLIST),$(subst /,,$(wildcard */)))
VERBOSE?=0
ifeq ($(VERBOSE),0)
MAKE_FLAGS=-s 2>/dev/null
else
MAKE_FLAGS=
endif

HIP_MAKEJ ?= 4



test:
	@for t in $(TESTS); do\
		echo "$(COLOR)--TESTING: $$t$(NOCOLOR)"; \
		cd $$t && make $@ $(MAKE_FLAGS) && cd .. || { echo "$(RED)BUILD FAILURE!!$(NOCOLOR)"; cd ..; }; \
	done


build:
	@for t in $(TESTS); do\
		echo "$(COLOR)--BUILDING: $$t$(NOCOLOR) j=$(HIP_MAKEJ)"; \
		cd $$t && make $(MAKE_FLAGS) -j $(HIP_MAKEJ) && cd .. || { echo "$(RED)BUILD FAILURE!!$(NOCOLOR)"; cd ..; }; \
	done

clean:
	@for t in $(TESTS); do\
		echo "$(COLOR)--CLEAN: $$t$(NOCOLOR)"; \
		cd $$t && make $@ $(MAKE_FLAGS) && cd .. || { echo "$(RED)CLEAN FAILURE!!$(NOCOLOR)"; cd ..; }; \
	done

perftest:
	@for t in $(TESTS); do\
		echo "$(COLOR)--PROFILING: $$t$(NOCOLOR)"; \
		rm -f $$t/*-perftest.log; \
		cd $$t && make $@ $(MAKE_FLAGS) && cd .. || { echo "$(RED)BUILD FAILURE!!$(NOCOLOR)"; cd ..; }; \
	done
	@echo "Sample-TestName,Iterations,Real(s),User(s),System(s)" > report.csv
	@grep -Rhs ".cmd" $(addsuffix /*-perftest.log,$(TESTS)) >> report.csv || true
	@echo "Summary saved in $(GREEN)report.csv$(NOCOLOR)"

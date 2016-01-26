FLAGS := -Wno-write-strings

all: hotspotex hotspotver

hotspotex: hotspotex.cpp
	g++ $(FLAGS) -o $@ $<

hotspotver: hotspotver.cpp
	g++ $(FLAGS) -o $@ $<

clean:
	rm hotspotex hotspotver


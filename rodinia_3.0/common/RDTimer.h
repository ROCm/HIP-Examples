#ifndef RODINIA_TIMER
#define RODINIA_TIMER

#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <string>
#include <float.h> 
#include <unistd.h>
#include <math.h>

typedef float SECS_t;

// Functions, can be used stand alone.

 // Returns the current system time in microseconds
inline long long get_time() 
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}

// Returns the number of seconds elapsed between the two specified times 
inline SECS_t elapsed_time(long long start_time, long long end_time) 
{
    return ((SECS_t) (end_time - start_time)) / ((SECS_t)(1000 * 1000)); 
}


class RDTimer
{
public:

  RDTimer(std::string inmsg = "") : accumulating(false), msg(inmsg), type("none"), start(0), end(0), elapsed(0.0f), accumulated(0.0f)  {;}
  virtual ~RDTimer() {;}
  
  virtual void Start() = 0;

  // time from last start to now.
  virtual SECS_t Stop() = 0;
  
  // accumulate time from previous start to now.
  virtual SECS_t Accumulate() = 0;

  SECS_t Time() const
  {
	if(accumulating==false) 
	  return elapsed;
	else
	  return accumulated;
  }
  
  void Reset()
  {
    accumulating = false;
    accumulated = 0.0f;
	elapsed = 0.0f;
  }

  void Reset(std::string inmsg)
  {
    msg = inmsg;
    Reset();
  }
  
  std::string Msg() const { return msg;}
  std::string Type() const { return type;}
  
  bool Accumulating() const {return accumulating;}
  
protected:
    bool accumulating;
	std::string msg;
	std::string type; 
	long long start;
	long long end;
	SECS_t elapsed;
	SECS_t accumulated;
};


class RDTimerCPU : public RDTimer
{
public:

  RDTimerCPU(std::string inmsg = "") : RDTimer(inmsg)
  {
	type = "CPU";
  }
  
  void Start()
  {
	start = get_time();
  }

  // time from last start to now.
  SECS_t Stop()
  {
     long long now = get_time();
	 elapsed = elapsed_time( start, now ); //in secs
	 return elapsed;
  }
  
  // accumulate time from previous start to now.
  SECS_t Accumulate()
  {
    accumulating = true;
    SECS_t interval = Stop();
	accumulated += interval;
	return accumulated;
  }
};

class RDTimerGPU : public RDTimer
{
public:

  RDTimerGPU(std::string inmsg = "") : RDTimer(inmsg)
  {
    type = "GPU";
 	hipEventCreate(&start);
	hipEventCreate(&stop); 
  }
 
  virtual ~RDTimerGPU()
  {
 	hipEventDestroy(start);
	hipEventDestroy(stop); 
  }
  
  void Start()
  {
	hipEventRecord(start,0);
  }

  // time from last start to now.
  SECS_t Stop()
  {
    hipDeviceSynchronize();
    hipEventRecord(stop,0);
    hipEventSynchronize(stop);
	
	//this returns time in msecs
	hipEventElapsedTime(&elapsed,start,stop);
	
	//convert to secs
	elapsed /= 1000.0f;
    return elapsed;
  }
  
  // accumulate time from previous start to now.
  SECS_t Accumulate()
  {
    accumulating = true;
    SECS_t interval = Stop();
	accumulated += interval;
	return accumulated;
  }
  
protected:
    
	hipEvent_t start;
	hipEvent_t stop;
};


class PerfSerializer
{
public:
	PerfSerializer( char* filename ) : success(true), fPtrRead(0), fPtrWrite(0), newFile(false)
	{
	
		if(  filename != (char*) 0 )
		{ 
			snprintf( fullName, sizeof(fullName), "%s%s", filename, ".perf");
			snprintf( fullNameTmp, sizeof(fullName), "%s%s", filename, ".perf.tmp");
			
			//file exists.
			if( access( fullName, F_OK ) == 0 )
			{
				fprintf(stderr, "File exists\n");
				
				fPtrRead = fopen(fullName, "r" );
				if( fPtrRead == 0 )
				{
					fprintf(stderr, "Could not open 1\n");
					throw "Could not open";
				}
			
				// write to a temp file, to avoid loss of data acquired so far
				// if program execution fails.
				fPtrWrite = fopen(fullNameTmp, "w" );
				if( fPtrWrite == 0 )
				{
					fprintf(stderr, "Could not open 2\n");
					fclose(fPtrRead);
					fPtrRead = 0;
					throw "Could not open";
				}
				
				newFile = false;
			}
			else
			{
				fPtrRead = 0;
				fPtrWrite = fopen(fullName, "w" );
				if( fPtrWrite == 0 )
				{
					fprintf(stderr, "Could not open 3\n");
					throw "Could not open";
				}
				
				newFile = true;
			}
			
			fprintf(stderr, "Opened file %s for performance log\n", fullName);
		}	
		else
		{
			throw "did not specify file";
		}
	}
	
	virtual ~PerfSerializer()
	{
		if(fPtrRead != 0)
		{
			fclose(fPtrRead);	
			fprintf(stderr, "Closed performance log 1\n");			
		}
		if(fPtrWrite != 0)
		{
			fclose(fPtrWrite);	
			fprintf(stderr, "Closed performance log 2\n");			
		}
		if(newFile==false)
		{
			fprintf(stderr, "COPYING FILE: %s to %s\n", fullNameTmp, fullName);
			char command[200];
			snprintf(command, sizeof(command), "mv %s %s", fullNameTmp, fullName);
			system(command);
		} 
	}
	
	virtual void Serialize(const RDTimer* const timer) = 0;
	
protected:
	bool success;
	FILE* fPtrRead;
	FILE* fPtrWrite;
	bool newFile;
	
	char fullName[100];
	char fullNameTmp[100];
};


class SimplePerfSerializer : public PerfSerializer
{
public:

	SimplePerfSerializer( char* filename ): PerfSerializer(filename), currLine(0), 
							read(-1), line(NULL), len(0), token(NULL){;}

	virtual void Serialize(const RDTimer* const timer)
	{
		int n = 0;
		SECS_t timesofar = 0.0f;
		
		if( newFile == false )
		{
			read = getline(&line, &len, fPtrRead);
						
			token = strtok( line, ",");
			if(token == NULL)
			{
				fprintf( stderr, "malformed line\n");
				throw "malformed line";	
			}
			
			if( strcmp( token, timer->Type().c_str() ) != 0 )
			{
				fprintf( stderr, "incompatible platform type\n");
				throw "incompatible platform type";
			}
			
			token = strtok (NULL, ",");
			if(token == NULL)
			{
				fprintf( stderr, "malformed line, second token\n");
				throw "malformed malformed line, second token";	
			}
			
			//ignore first space
			if( strcmp( &token[1], timer->Msg().c_str() ) != 0 )
			{
				fprintf( stderr, "incompatible message:%s,%s\n", token, timer->Msg().c_str());
				throw "incompatible incompatible message";
			}
			
			token = strtok (NULL, ",");
			if(token == NULL)
			{
				fprintf( stderr, "malformed line, third token\n");
				throw "malformed line, third token";	
			}
			
			if( strcmp( &token[1], timer->Accumulating() ? "ACCUM" : "ONEOFF" ) != 0 )
			{
				fprintf( stderr, "incompatible timer type\n");
				throw "incompatible incompatible timer type";
			}
			
			token = strtok (NULL, ",");
			if(token == NULL)
			{
				fprintf( stderr, "malformed line forth token\n");
				throw "malformed line forth token";	
			}
			
			timesofar = atof(token); //average
			token = strtok (NULL, ",");
			if(token == NULL)
			{
				fprintf( stderr, "INFO: number of trials so far not found. Old format. Will update\n");	
			}			
			
			if(token != NULL) 
			{
				n = atoi(token);
			}
			else
			{
				n=1;
			}
			
			timesofar *= n;  // sum so far
		}
		
		// new average
		timesofar = (timer->Time() + timesofar)/ (n+1);
		
		//NOTE: the spaces below matter, if you change the formating you will need to change the
		// parsing.
		fprintf(fPtrWrite, "%s, %s, %s, %f, %d\n", 
			timer->Type().c_str(), 
			timer->Msg().c_str(), 
			timer->Accumulating() ? "ACCUM" : "ONEOFF",
			timesofar,
			n+1);
	}	
	
protected:
	int currLine;
	ssize_t read;
	char * line = NULL;
	size_t len = 0;	
	char* token;
			
};

#endif // #define RODINIA_TIMER

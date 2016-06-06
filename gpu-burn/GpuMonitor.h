/*
 * Public domain.  No warranty.
 * Ville Timonen 2013
 * edited by Timmy Liu for HIP API 01/2016
 */

#ifndef GPUBURN_GPUMONITOR_H_
#define GPUBURN_GPUMONITOR_H_

// ---------------------------------------------------------------------------
namespace gpuburn {

/**
 * The GpuMonitor provides a generic interface to access common
 * GPU hardware data
 */
class GpuMonitor {
    public:
        virtual ~GpuMonitor() {};


        /**
         * Retreive the current temperature in degrees Celcius
         * for this device.
         */
        virtual float getTemperature() = 0;

        /**
         * Retreive the current temperature in degrees Celcius
         * for this device.
         */
        virtual int getId() { return mId; }

    protected:
        GpuMonitor(int id) : mId(id) {};

    private:
        int mId;
};

}; // namespace gpuburn

// ---------------------------------------------------------------------------

#endif // GPUBURN_GPUMONITOR_H_

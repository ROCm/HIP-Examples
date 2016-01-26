#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>

#include "bmp.h"

using std::ofstream;
using std::sort;
using std::cout;
using std::endl;
using std::swap;
using std::ios;

// ----------------------------------------------------------------
// Local Function Prototypes
// ----------------------------------------------------------------

static void doWrite(ofstream &out, int value);
static void doWrite(ofstream &out, short value);
static void doWrite(ofstream &out, const Color &theColor);

// ----------------------------------------------------------------
// Class: Bmp
// ----------------------------------------------------------------

Bmp::Bmp(int width, int height) : mWidth(width), mHeight(height)
{
   // Initialize the entire bitmap to the default color (white)

   for(int i = 0; i < mHeight; i ++)
   {
      ColorRow row;

      for(int j = 0; j < mWidth; j ++)
         row.push_back(Color());

      mImage.push_back(row);
   }
}

// ----------------------------------------------------------------
// Draw a pixel at (x, y) with color: (red, green, blue)
// ----------------------------------------------------------------

void Bmp::setPixel(
   int x,
   int y,
   unsigned char red,
   unsigned char green,
   unsigned char blue
)
{
   bool doSet = true;

   if(x < 0 || x >= mWidth)
   {
      cout << "Invalid value: " << x <<
         " (expected: 0 <= x < " << mWidth << ")" << endl;

      doSet = false;
   }

   if(y < 0 || y >= mHeight)
   {
      cout << "Invalid value: " << y <<
         " (expected: 0 <= y < " << mHeight << ")" << endl;

      doSet = false;
   }

   if(doSet)
      mImage[y][x] = Color(red, green, blue);
}

// ----------------------------------------------------------------
// Draw a pixel at (x, y) with color: theColor
// ----------------------------------------------------------------

void Bmp::setPixel(
   int x,
   int y,
   const Color &theColor  
)
{
   setPixel(x, y, theColor.mRed, theColor.mGreen, theColor.mBlue);
}

// ----------------------------------------------------------------
// Draw a rectangle at (x, y) with the specified width and
// height using color: (red, green, blue)
// ----------------------------------------------------------------

void Bmp::drawRectangle(
   int x,
   int y,
   int width,
   int height,
   unsigned char red,
   unsigned char green,
   unsigned char blue
)
{    
   drawLine(x,         y,          x + width, y,          red, green, blue);
   drawLine(x + width, y,          x + width, y + height, red, green, blue);
   drawLine(x + width, y + height, x,         y + height, red, green, blue);
   drawLine(x,         y + height, x,         y,          red, green, blue);
}

// ----------------------------------------------------------------
// Draw a rectangle at (x, y) with the specified width and
// height using color: theColor
// ----------------------------------------------------------------

void Bmp::drawRectangle(
   int x,
   int y,
   int width,
   int height,
   const Color &theColor                       
)
{
   drawRectangle(
      x, y, width, height,
      theColor.mRed, theColor.mGreen, theColor.mBlue
   );
}

// ----------------------------------------------------------------
// Fill a rectangle at (x, y) with the specified width and
// height using color: (red, green, blue)
// ----------------------------------------------------------------

void Bmp::fillRectangle(
   int x,
   int y,
   int width,
   int height,
   unsigned char red,
   unsigned char green,
   unsigned char blue
)
{
   for(int i = 0; i < width; i ++)
   {
      drawLine(
         x + i, y, x + i, y + height, red, green, blue
      );
   }
}

// ----------------------------------------------------------------
// Fill a rectangle at (x, y) with the specified width and
// height using color: theColor
// ----------------------------------------------------------------

void Bmp::fillRectangle(
   int x,
   int y,
   int width,
   int height,
   const Color &theColor                       
)
{
   fillRectangle(
      x, y, width, height,
      theColor.mRed, theColor.mGreen, theColor.mBlue
   );
}

// ----------------------------------------------------------------
// Fill a polygon using color: (red, green, blue),
// expects 'points' to contain an even number of values
// ----------------------------------------------------------------

void Bmp::fillPolygon(
   const    vector<int> &points,
   unsigned char        red,
   unsigned char        green,
   unsigned char        blue
)
{
   int xMin = 0;
   int yMin = 0;
   int xMax = 0;
   int yMax = 0;

   for(int i = 0, n = points.size(); i < n; i += 2)
   {
      int x = points[i];
      int y = points[i + 1];
      
      if(i == 0)
      {
         xMin = xMax = x;
         yMin = yMax = y;
      }
      else
      {
         if(x < xMin)      xMin = x;
         else if(x > xMax) xMax = x;

         if(y < yMin)      yMin = y;
         else if(y > yMax) yMax = y;
      }
   }

   typedef vector<int>     YValues;
   typedef vector<YValues> Buffer;

   int width = (xMax - xMin) + 1;

   Buffer theBuffer;

   for(int i = 0; i < width; i ++)
      theBuffer.push_back(YValues());

   for(int i = 2, n = points.size(); i < n; i += 2)
   {
      int x0 = points[i - 2];
      int y0 = points[i - 1];

      int x1 = points[i];
      int y1 = points[i + 1];

      if(x0 == x1)
      {
         theBuffer[x0 - xMin].push_back(y0);
         theBuffer[x0 - xMin].push_back(y1);
      }
      else 
      {
         double delta = (y1 - y0) / (double)(x1 - x0);
         int    start = 0;
         int    stop  = 0;
         double y     = 0.0;
         
         if(x0 < x1)
         {
            y     = y0;
            start = x0;
            stop  = x1;
         }
         else
         {
            y     = y1;
            start = x1;
            stop  = x0;
         }

         for(int j = start; j < stop; j ++)
         {
            int yValue = (int)(y + 0.5);

            theBuffer[j - xMin].push_back(yValue);

            y += delta;
         }
      }
   }

   for(int i = 0, n = theBuffer.size(); i < n; i ++)
      sort(theBuffer[i].begin(), theBuffer[i].end());

   for(int i = 0, n = theBuffer.size(); i < n; i ++)
   {
      for(int j = 1, nn = theBuffer[i].size(); j < nn; j += 2)
      {
         drawLine(
            xMin + i,
            theBuffer[i][j],
            xMin + i,
            theBuffer[i][j - 1],
            red, green, blue
         );
      }
   }
}

// ----------------------------------------------------------------
// Fill a polygon using color: theColor
// ----------------------------------------------------------------

void Bmp::fillPolygon(const vector<int> &points, const Color &theColor)
{
   fillPolygon(points, theColor.mRed, theColor.mGreen, theColor.mBlue);
}

// ----------------------------------------------------------------
// Draw a line from (x0, y0) to (x1, y1) using
// color: (red, green, blue)
// ----------------------------------------------------------------

void Bmp::drawLine(
   int x0, int y0,
   int x1, int y1, 
   unsigned char red,
   unsigned char green,
   unsigned char blue
)
{
   int xDelta = (x1 - x0);
   int yDelta = (y1 - y0);

   if(xDelta == 0)
   {
      // A vertical line

      if(y0 > y1)
         swap(y0, y1);

      for(int y = y0; y <= y1; y ++)
         setPixel(x0, y, red, green, blue);
   }
   else if(yDelta == 0)
   {
      // A horizontal line

      if(x0 > x1)
         swap(x0, x1);

      for(int x = x0; x <= x1; x ++)
         setPixel(x, y0, red, green, blue);
   }
   else
   {
      setPixel(x0, y0, red, green, blue);

      int xStep = (xDelta < 0 ? -1 : 1);
      int yStep = (yDelta < 0 ? -1 : 1);

      xDelta = abs(xDelta) / 2;
      yDelta = abs(yDelta) / 2;

      if(xDelta >= yDelta)
      {
         int error = yDelta - 2 * xDelta;
 
         while(x0 != x1)
         {
            if(error >= 0 && (error || xStep > 0))
            {
               error -= xDelta;
               y0    += yStep;
            }

            error += yDelta;
            x0    += xStep;
 
            setPixel(x0, y0, red, green, blue);
         }
      }
      else
      {
         int error = xDelta - 2 * yDelta;
 
         while(y0 != y1)
         {
            if(error >= 0 && (error || yStep > 0))
            {
               error -= yDelta;
               x0    += xStep;
            }

            error += xDelta;
            y0    += yStep;
 
            setPixel(x0, y0, red, green, blue);
         }
      }
   }
}

// ----------------------------------------------------------------
// Draw a line from (x0, y0) to (x1, y1) using color: theColor
// ----------------------------------------------------------------

void Bmp::drawLine(
   int x0, int y0,
   int x1, int y1, 
   const Color &theColor                       
)
{
   drawLine(
      x0, y0, x1, y1, theColor.mRed, theColor.mGreen, theColor.mBlue
   );
}

// ----------------------------------------------------------------
// Draw a multi-part line, given points:
//
//   { x0, y0, x1, y1, x2, y2, ... }
//
// A line will be drawn from: (x0, y0) to (x1, y1), from (x1, y1)
// to (x2, y2), etc.  Color will be: (red, green, blue)
// ----------------------------------------------------------------

void Bmp::drawPolyline(
   const vector<int> &points,
   unsigned char      red,
   unsigned char      green,
   unsigned char      blue
)
{
   int xPrevious = 0;
   int yPrevious = 0;

   for(int i = 0, n = points.size(); i < n; i += 2)
   {
      if(i == 0)
      {
         xPrevious = points[i];
         yPrevious = points[i + 1];
      }
      else
      {
         int x = points[i];
         int y = points[i + 1];

         drawLine(xPrevious, yPrevious, x, y, red, green, blue);

         xPrevious = x;
         yPrevious = y;
      }
   }
}

void Bmp::drawPolyline(const vector<int> &points, const Color &theColor)
{
   drawPolyline(points, theColor.mRed, theColor.mGreen, theColor.mBlue);
}

// ----------------------------------------------------------------
// The bitmap width
// ----------------------------------------------------------------

int Bmp::getWidth() const
{
   return(mWidth);
}

// ----------------------------------------------------------------
// The bitmap height
// ----------------------------------------------------------------

int Bmp::getHeight() const
{
   return(mHeight);
}

// ----------------------------------------------------------------
// Write our bitmap to: fileName, return true on succes,
// on error: populate errMsg, return false
// ----------------------------------------------------------------

bool Bmp::write(string &fileName, string &errMsg) const
{
   ofstream out(fileName.c_str(), ios::binary);

   if(out.fail())
   {
      errMsg = "Could not open: [" + fileName + "]";
      return(false);
   }

   // Header sizes ...

   const int BMP_FILE_HEADER_SIZE = 14;
   const int BMP_INFO_HEADER_SIZE = 40;

   // ----------------------------------------------
   // The bmp file header
   // ----------------------------------------------

   out.put('B');
   out.put('M');

   int fileSize =
      mWidth * mHeight * 3 +
      BMP_FILE_HEADER_SIZE + BMP_INFO_HEADER_SIZE;

   doWrite(out, fileSize);
  
   short reserved = 0;
   doWrite(out, reserved);
   doWrite(out, reserved);
  
   int offset = BMP_FILE_HEADER_SIZE + BMP_INFO_HEADER_SIZE;
   doWrite(out, offset);

   // ----------------------------------------------
   // The bmp information header
   // ----------------------------------------------

   int headerSize = BMP_INFO_HEADER_SIZE;
   doWrite(out, headerSize);

   doWrite(out, mWidth);
   doWrite(out, mHeight);

   short colorPlanes = 1;
   doWrite(out, colorPlanes);

   short bitsPerPixel = 24;
   doWrite(out, bitsPerPixel);

   int zero = 0;

   for(int i = 0; i < 6; i ++)
      doWrite(out, zero);
 
   for(int i = 0; i < mHeight; i ++)  
      for(int j = 0; j < mWidth; j ++)   
         doWrite(out, mImage[i][j]);
  
   out.close();

   return(true);
}

// ----------------------------------------------------------------
// Local Functions
// ----------------------------------------------------------------

static void doWrite(ofstream &out, int value)
{
   out.write((const char *)&value, sizeof(int));
}

static void doWrite(ofstream &out, short value)
{
   out.write((const char *)&value, sizeof(short));
}

static void doWrite(ofstream &out, const Color &theColor)
{
   out.write((const char *)&theColor.mBlue,  sizeof(unsigned char));    
   out.write((const char *)&theColor.mGreen, sizeof(unsigned char));
   out.write((const char *)&theColor.mRed,   sizeof(unsigned char));
}
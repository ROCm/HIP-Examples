#include <iostream>
#include <string>
#include <cmath>

#include "bmp.h"

using namespace std;

static const int WIDTH  = 4;
static const int HEIGHT = 4;

static void showCheckerBoard(Bmp &bmp)
{
//   Color c1 = Color(235, 235, 235);
   Color c2 = Color::white();
	Color c1 = Color(0, 0, 0);
//	Color c2 = Color(255, 255, 255);
	Color c3 = Color(-1,-1,-1);
	Color c4 = Color(1, 1, 1);

	//big checkerBoard
   for(int i = 0; i < HEIGHT; i ++)
   {
      for(int j = 0; j < WIDTH; j ++)
      {
//         if((i / 30) % 2 == (j / 30) % 2)
		   if((i/2)%2 == (j/2)%2)
            bmp.setPixel(j, i, c1);
			//bmp.setPixel(j, i, c3);
         else
            bmp.setPixel(j, i, c2);
			//bmp.setPixel(j, i, c4);
      }
   }


/*	for(int i = 0; i < HEIGHT; i ++)
   {
      for(int j = 0; j < WIDTH; j ++)
      {
         if(j % 2 == 0)
            bmp.setPixel(j, i, c1);

         else
            bmp.setPixel(j, i, c2);
      }
   }
*/
}

static void showRectangles(Bmp &bmp)
{
   bmp.drawRectangle(0, 0, WIDTH - 1, HEIGHT - 1, Color::cyan());

   bmp.fillRectangle(20, 20, 40, 60, Color::yellow());
   bmp.drawRectangle(20, 20, 40, 60, Color::black());

   int xStep = 4;
   int yStep = 4;

   int nRectangles = 10;
   double dRed     = 0.0;
   double dGreen   = 255.0;
   double dBlue    = 0.0;

   double stepValue = (255.0 / (nRectangles - 1));

   double dRedStep   = 0.0;
   double dGreenStep = -stepValue;
   double dBlueStep  = stepValue;

   for(int i = 0; i < nRectangles; i ++)
   {
      Color theColor(
         (unsigned char)(dRed   + 0.5),
         (unsigned char)(dGreen + 0.5),
         (unsigned char)(dBlue  + 0.5)
      );

      bmp.drawRectangle(
         70 + xStep * i, 50 - yStep * i, 80, 60, theColor
      );

      dRed   += dRedStep;
      dGreen += dGreenStep;
      dBlue  += dBlueStep;
   }
}

static double degreesToRadians(double degrees)
{
   double radians = (3.14159265 / 180.0) * degrees;

   return(radians);
}

static void showLines(Bmp &bmp)
{
   int xCenter = WIDTH  / 2;
   int yCenter = HEIGHT / 2;

   double degrees = 0.0;
   double radius  = 75.0;

   while(degrees < 360.0)
   {
      double radians = degreesToRadians(degrees);
      double x       = radius * cos(radians);
      double y       = radius * sin(radians);

      bmp.drawLine(
         xCenter, yCenter,
         xCenter + (int)x, yCenter + (int)y,
         Color::red()
      );

      degrees += 10.0;
   }
}

static void showPolygons(Bmp &bmp)
{
   vector<int> points;

   double step = 2 * (360.0 / 5.0);

   int xOffset = bmp.getWidth() / 2 - 50;
   int yOffset = 125  + bmp.getHeight() / 2;

   double angle = 0.0;
   double radius = 50.0;

   for(int i = 0; i < 6; i ++)
   {
      double radians = degreesToRadians(angle);

      double x = xOffset + radius * cos(radians);
      double y = yOffset + radius * sin(radians);

      points.push_back((int)(x + 0.5));
      points.push_back((int)(y + 0.5));

      angle += step;

      if(angle >= 360.0)
         angle -= 360.0;
   }

   bmp.fillPolygon(points, Color::black());
}

int main()
{
   Bmp image(WIDTH, HEIGHT);

   bool doCheckerBoard = true;
/*   bool doRectangles   = true;
   bool doLines        = true;
   bool doPolygons     = true;
*/
   if(doCheckerBoard)
   {
      showCheckerBoard(image);
   }

/*
   if(doRectangles)
   {
      showRectangles(image);
   }

   if(doLines)
   {
      showLines(image);
   }

   if(doPolygons)
   {
      showPolygons(image);
   }
*/
   string errMsg;
   string fileName = "4.bmp";

   if(!image.write(fileName, errMsg))
   {
      cout << errMsg << endl;
   }
   else
   {
      cout << "Successfully wrote file: [" << fileName << "]" << endl;
   }

   return(0);
}
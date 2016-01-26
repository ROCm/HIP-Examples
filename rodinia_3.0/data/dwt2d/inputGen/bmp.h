#ifndef _BMP_H_
#  define _BMP_H_

   #include <vector>
   #include <string>

   using std::string;
   using std::vector;

   struct Color
   {   
      Color(
         unsigned char red,
         unsigned char green,
         unsigned char blue
      ) :
         mRed(red), mGreen(green), mBlue(blue)
      {
      }

      Color() : mRed(255), mGreen(255), mBlue(255)
      {
      }

      static Color red()     { return(Color(255, 0,     0)); }
      static Color green()   { return(Color(0,   255,   0)); }
      static Color blue()    { return(Color(0,   0,   255)); }

      static Color cyan()    { return(Color(0,   255, 255)); }
      static Color magenta() { return(Color(255,   0, 255)); }
      static Color yellow()  { return(Color(255, 255,   0)); }
      static Color black()   { return(Color(0,   0,     0)); }

      static Color white() { return(Color()); }

      unsigned char mRed;
      unsigned char mGreen;
      unsigned char mBlue;
   };

   typedef vector<Color> ColorRow;

   class Bmp
   {
      public:

         Bmp(int width, int height);   

         void setPixel(
            int x,
            int y,
            unsigned char red,
            unsigned char green,
            unsigned char blue
         );

         void setPixel(
            int x,
            int y,
            const Color &theColor          
         );

         void drawRectangle(
            int x,
            int y,
            int width,
            int height,
            unsigned char red,
            unsigned char green,
            unsigned char blue
         );

         void drawRectangle(
            int x,
            int y,
            int width,
            int height,
            const Color &theColor                       
         );

         void fillRectangle(
            int x,
            int y,
            int width,
            int height,
            unsigned char red,
            unsigned char green,
            unsigned char blue
         );

         void fillRectangle(
            int x,
            int y,
            int width,
            int height,
            const Color &theColor                       
         );

         void fillPolygon(
            const vector<int> &points,
            unsigned char red,
            unsigned char green,
            unsigned char blue
         );

         void fillPolygon(
            const vector<int> &points,
            const Color &theColor                       
         );

         void drawLine(
            int x0, int y0,
            int x1, int y1, 
            unsigned char red,
            unsigned char green,
            unsigned char blue
         );

         void drawLine(
            int x0, int y0,
            int x1, int y1, 
            const Color &theColor                       
         );

         void drawPolyline(
            const vector<int> &points,
            unsigned char red,
            unsigned char green,
            unsigned char blue
         );

         void drawPolyline(
            const vector<int> &points,
            const Color &theColor
         );

         int getWidth()  const;
         int getHeight() const;

         bool write(string &fileName, string &errMsg) const;
   
      private:

         int mWidth;
         int mHeight;
         vector<ColorRow> mImage;
   };

#endif
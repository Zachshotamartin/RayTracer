#ifndef RAY_H
#define RAY_H


// Class built upon the vec3 class
class ray {
    
  public:
    
    
    // Constructors
    ray() {}
    ray(const point3& origin, const vec3& direction) : origin(origin), direction(direction) {}
    
    
    // Class Getters
    const point3& o() const  { return origin; }
    const vec3& d() const { return direction; }
    
    
    // Class Functions
    point3 at(double t) const {
        return origin + t*direction;
    }
    
    
  private:
    
    
    // Class Variables
    point3 origin;
    vec3 direction;
};

#endif

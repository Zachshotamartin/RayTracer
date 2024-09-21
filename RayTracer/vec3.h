#ifndef vec3_h
#define vec3_h


class vec3 {
    public:
        
    
        // Class Variables
        double v[3];
        
        
        // Constructors
        vec3(): v{0, 0, 0} {}
        vec3(double x, double y, double z): v{x, y, z} {}
        
        
        // Class Getters
        double x() const {
            return v[0];
        }
        double y() const {
            return v[1];
        }
        double z() const {
            return v[2];
        }
    
        
        // Class Functions
        double length() const {
            return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        }
        double dot() const {
            return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        }
        
    
        // Class Operator Overloads
        vec3 operator-() const {
            return vec3(-v[0], -v[1], -v[2]);
        }
        double operator[](int i) const {
            return v[i];
            
        }
        double& operator[](int i) {
            return v[i];
        }
        vec3& operator+=(const vec3& v2) {
            v[0] += v2[0];
            v[1] += v2[1];
            v[2] += v2[2];
            return *this;
        }
        vec3& operator-=(const vec3& v2) {
            v[0] -= v2[0];
            v[1] -= v2[1];
            v[2] -= v2[2];
            return *this;
        }
        vec3& operator*=(double a) {
            v[0] *= a;
            v[1] *= a;
            v[2] *= a;
            return *this;
        }
        vec3& operator/=(double a) {
            v[0] /= a;
            v[1] /= a;
            v[2] /= a;
            return *this;
        }
};


// Helps to differentiat a direction vector from a point vector
using point3 = vec3;


// Helper Utility Operator Overloads
inline vec3 operator+(const vec3& v1, const vec3& v2) {
    return vec3(v1.v[0]+v2.v[0], v1.v[1]+v2.v[1], v1.v[2]+v2.v[2]);
}
inline vec3 operator-(const vec3& v1, const vec3& v2) {
    return vec3(v1.v[0]-v2.v[0], v1.v[1]-v2.v[1], v1.v[2]-v2.v[2]);
}
inline vec3 operator*(const vec3& v1, double a) {
    return vec3(a*v1.v[0], a*v1.v[1], a*v1.v[2]);
}
inline vec3 operator*(double a, const vec3& v1) {
    return vec3(a*v1.v[0], a*v1.v[1], a*v1.v[2]);
}
inline vec3 operator/(const vec3& v1, double a) {
    return vec3(v1.v[0]/a, v1.v[1]/a, v1.v[2]/a);
}


// Helper Utility Functions
inline vec3 to_unit_vector(const vec3& v) {
    return v / v.length();
}
inline double dot(const vec3& v1, const vec3& v2) {
    return v1.v[0]*v2.v[0] + v1.v[1]*v2.v[1] + v1.v[2]*v2.v[2];
}
inline vec3 cross(const vec3& v1, const vec3& v2) {
    return vec3(v1.v[1]*v2.v[2] - v1.v[2]*v2.v[1],
                v1.v[2]*v2.v[0] - v1.v[0]*v2.v[2],
                v1.v[0]*v2.v[1] - v1.v[1]*v2.v[0]
                );
}

#endif /* vec3_h */

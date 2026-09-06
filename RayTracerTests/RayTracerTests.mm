#include "../RayTracer/headers_and_constants.h"
#import <XCTest/XCTest.h>

@interface RayTracerTests : XCTestCase
@end
@implementation RayTracerTests
- (void)testTranslatedBoxNormal {
    auto mat = std::make_shared<lambertian>(color(1, 1, 1));
    cube box(point3(2, 3, 4), vec3(2, 2, 2), vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1), mat);
    hit_record rec;
    XCTAssertTrue(box.hit(ray(point3(5, 3, 4), vec3(-1, 0, 0)), interval(0.001, infinity), rec));
    XCTAssertEqualWithAccuracy(rec.t, 2, 1e-9);
    XCTAssertEqualWithAccuracy(rec.normal.x(), 1, 1e-9);
    XCTAssertEqualWithAccuracy(rec.normal.z(), 0, 1e-9);
}
- (void)testInsideBoxExit {
    cube box(point3(), vec3(2, 2, 2), vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1),
             std::make_shared<lambertian>(color(1, 1, 1)));
    hit_record rec;
    XCTAssertTrue(box.hit(ray(point3(), vec3(1, 0, 0)), interval(0.001, infinity), rec));
    XCTAssertEqualWithAccuracy(rec.t, 1, 1e-9);
    XCTAssertFalse(rec.front_face);
}
- (void)testPointLightHasFiniteVisibility {
    hittable_list world;
    world.add(std::make_shared<sphere>(point3(0, 3, 0), 0.5,
                                       std::make_shared<lambertian>(color(1, 1, 1))));
    PointLight light(point3(0, 1, 0), color(1, 1, 1), 1);
    sampler rng;
    XCTAssertTrue(light_list::visible(point3(), light.sample(point3(), rng), world));
}
@end

#import <XCTest/XCTest.h>

@interface RayTracerUITests : XCTestCase
@end
@implementation RayTracerUITests
- (void)setUp {
    self.continueAfterFailure = NO;
}
- (void)testViewerControls {
    XCUIApplication *app = [[XCUIApplication alloc] init];
    // Metal drawables can be unavailable when the automation window is occluded.
    app.launchEnvironment = @{@"SDL_RENDER_DRIVER" : @"software"};
    app.launchArguments =
        @[ @"--scene", @"studio", @"--width", @"160", @"--samples", @"100000", @"--threads", @"2" ];
    [app launch];
    [app activate];
    XCUIElement *window = app.windows.firstMatch;
    XCTAssertTrue([window waitForExistenceWithTimeout:30]);
    [window click];
    [window typeKey:@" " modifierFlags:0];
    NSPredicate *paused = [NSPredicate predicateWithFormat:@"label CONTAINS %@", @"PAUSED"];
    [self expectationForPredicate:paused evaluatedWithObject:window handler:nil];
    [self waitForExpectationsWithTimeout:10 handler:nil];
    [window typeKey:@"2" modifierFlags:0];
    NSPredicate *field = [NSPredicate predicateWithFormat:@"label CONTAINS %@", @"field"];
    [self expectationForPredicate:field evaluatedWithObject:window handler:nil];
    [self waitForExpectationsWithTimeout:10 handler:nil];
    [app terminate];
}
@end

#import <XCTest/XCTest.h>
@interface RayTracerUITestsLaunchTests : XCTestCase
@end
@implementation RayTracerUITestsLaunchTests
- (void)setUp {
    self.continueAfterFailure = NO;
}
- (void)testLaunchShowsFrame {
    XCUIApplication *app = [[XCUIApplication alloc] init];
    app.launchEnvironment = @{@"SDL_RENDER_DRIVER" : @"software"};
    app.launchArguments = @[ @"--width", @"160", @"--samples", @"16", @"--threads", @"2" ];
    [app launch];
    [app activate];
    XCTAssertTrue([app.windows.firstMatch waitForExistenceWithTimeout:10]);
    XCTAttachment *attachment = [XCTAttachment attachmentWithScreenshot:app.screenshot];
    attachment.name = @"Rendered frame";
    attachment.lifetime = XCTAttachmentLifetimeKeepAlways;
    [self addAttachment:attachment];
    [app terminate];
}
@end

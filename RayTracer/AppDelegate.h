//
//  AppDelegate.h
//  RayTracer
//
//  Created by Zachary Martin on 9/17/24.
//

#import <Cocoa/Cocoa.h>
#import <CoreData/CoreData.h>

@interface AppDelegate : NSObject <NSApplicationDelegate>

@property (readonly, strong) NSPersistentContainer *persistentContainer;


@end


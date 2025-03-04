#pragma once

struct PointXYZCustom
{
    PCL_ADD_POINT4D;  // Adds the basic XYZ fields
    float background;
    float smooth;
    float grass;
    float rough;
    float lowVeg;
    float highVeg;
    float sky;
    float obstacle;
    float intensity;
    union
    {
        struct
        {
            uint8_t b;
            uint8_t g;
            uint8_t r;
            uint8_t _unused;
        };
        float rgb;
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Ensures proper alignment

    PointXYZCustom() 
        : x(0), y(0), z(0),   // Assuming PCL_ADD_POINT4D defines x, y, and z
        background(0),
        smooth(0),
        grass(0),
        rough(0),
        lowVeg(0),
        highVeg(0),
        sky(0),
        obstacle(0),
        intensity(0.0),
        rgb(0)
    {
        b = 202;
        g = 187;
        r = 201;
        _unused = 0;
    }
} EIGEN_ALIGN16;  // Forces SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZCustom,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, background, background)
                                   (float, smooth, smooth)
                                   (float, grass, grass)
                                   (float, rough, rough)
                                   (float, lowVeg, lowVeg)
                                   (float, highVeg, highVeg)
                                   (float, sky, sky)
                                   (float, obstacle, obstacle)
                                   (float, intensity, intensity)
                                   (float, rgb, rgb)
)


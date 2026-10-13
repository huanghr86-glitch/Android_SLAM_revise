#include  "utils/Communicator.h"
#include <cctype>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <sstream>

#include <android/log.h>
#define DEBUG_INFO(...) ::__android_log_print(ANDROID_LOG_INFO, "[Android SLAM DEBUG]", __VA_ARGS__)

namespace android_slam
{
    static std::string trimAscii(std::string s)
    {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
            s.erase(s.begin());
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
            s.pop_back();
        return s;
    }

    static bool parsePort(const std::string& text, int& port)
    {
        try
        {
            size_t pos = 0;
            int value = std::stoi(text, &pos);
            while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
                ++pos;
            if (pos != text.size() || value <= 0 || value > 65535)
                return false;
            port = value;
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    static bool sendAll(int socket_fd, const void* buffer, size_t length)
    {
        const char* data = static_cast<const char*>(buffer);
        size_t sent = 0;
        while (sent < length)
        {
            ssize_t n = ::send(socket_fd, data + sent, length - sent, 0);
            if (n < 0)
            {
                if (errno == EINTR)
                    continue;
                return false;
            }
            if (n == 0)
                return false;
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    Communicator::Communicator()
    {
        DEBUG_INFO("[Android Slam App Info] Communicator is built.");
    }

    void Communicator::Run(float x, float y, float z, std::string ip, std::string port)
    {
        int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd == -1)
        {
            //std::cout << "socket 创建失败：" << std::endl;
            DEBUG_INFO("[Android Slam App Info] Socket create failed.");
            return;
        }
        DEBUG_INFO("[Android Slam App Info] Socket create succesfully.");
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        //addr.sin_port = htons(8051);
        //addr.sin_addr.s_addr = inet_addr("10.134.114.59");
        ip = trimAscii(ip);
        port = trimAscii(port);
        DEBUG_INFO("[Comm] trying connect ip='%s' port='%s'", ip.c_str(), port.c_str());
        int port_number = 0;
        if (!parsePort(port, port_number))
        {
            DEBUG_INFO("[Comm] invalid port string: '%s'", port.c_str());
            close(socket_fd);
            return;
        }
        addr.sin_port = htons(port_number);
        uint32_t parsed_ip = inet_addr(ip.c_str());
        if (parsed_ip == INADDR_NONE)
        {
            DEBUG_INFO("[Comm] invalid IP string: '%s'", ip.c_str());
            close(socket_fd);
            return;
        }
        addr.sin_addr.s_addr = parsed_ip;

        int res = connect(socket_fd, (struct sockaddr *)&addr, sizeof(addr));
        if (res == -1)
        {
            DEBUG_INFO("[Android Slam App Info] Socket connect failed. errno=%d reason=%s", errno, strerror(errno));
            return;
        }
        DEBUG_INFO("[Android Slam App Info] Socket connect successfully.");

        float commu_data[4] = {x, y, z, 1.0};
        //std::cout << commu_data[0] << " " << commu_data[1] << " " << commu_data[2] << " " << commu_data[3] << std::endl;
        DEBUG_INFO("[Android Slam App Info] Socket send data: x: %f, y: %f, z: %f.", x, y, z);
        sendAll(socket_fd, commu_data, sizeof(commu_data));
        close(socket_fd);
        //usleep(100000);
        return;
    }

    void Communicator::Run(TrackingResult track_res, std::string ip, std::string port)
    {
        int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd == -1)
        {
            //std::cout << "socket 创建失败：" << std::endl;
            DEBUG_INFO("[Android Slam App Info] Socket create failed.");
            return;
        }
        DEBUG_INFO("[Android Slam App Info] Socket create succesfully.");
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        //addr.sin_port = htons(9090);
        //addr.sin_addr.s_addr = inet_addr("192.168.43.172");
        ip = trimAscii(ip);
        port = trimAscii(port);
        DEBUG_INFO("[Comm] trying connect ip='%s' port='%s'", ip.c_str(), port.c_str());
        int port_number = 0;
        if (!parsePort(port, port_number))
        {
            DEBUG_INFO("[Comm] invalid port string: '%s'", port.c_str());
            close(socket_fd);
            return;
        }
        addr.sin_port = htons(port_number);
        uint32_t parsed_ip = inet_addr(ip.c_str());
        if (parsed_ip == INADDR_NONE)
        {
            DEBUG_INFO("[Comm] invalid IP string: '%s'", ip.c_str());
            close(socket_fd);
            return;
        }
        addr.sin_addr.s_addr = parsed_ip;

        int res = connect(socket_fd, (struct sockaddr *)&addr, sizeof(addr));

        if (res == -1)
        {
            DEBUG_INFO("[Android Slam App Info] Socket connect failed. errno=%d reason=%s", errno, strerror(errno));
            return;
        }
        DEBUG_INFO("[Android Slam App Info] Socket connect successfully.");

        const size_t point_count = track_res.trajectory.size();
        if (point_count > (static_cast<size_t>(UINT32_MAX) / 3u))
        {
            DEBUG_INFO("[Android Slam App Info] too many trajectory points to send.");
            close(socket_fd);
            return;
        }

        const uint32_t size = static_cast<uint32_t>(point_count) * 3u;
        sendAll(socket_fd, &size, sizeof(uint32_t));
        DEBUG_INFO("[Android Slam App Info] point count = %lu", (unsigned long) point_count);

        // 保持既有线上协议：先发 float 数量，再发 x, z, y 顺序的轨迹点。
        std::vector<float> commu_data;
        commu_data.reserve(point_count * 3u);
        for (const auto& data : track_res.trajectory)
        {
            commu_data.push_back(data.x + start_pos[0]);
            commu_data.push_back(data.z + start_pos[1]);
            commu_data.push_back(data.y + start_pos[2]);
        }
        sendAll(socket_fd, commu_data.data(), sizeof(float) * commu_data.size());
        DEBUG_INFO("[Android Slam App Info] sent %lu floats", (unsigned long) commu_data.size());
        close(socket_fd);
        //usleep(100000);
        return;
    }

    void Communicator::getStartPos(std::string start_str)
    {
        std::istringstream iss(start_str);
        float value = 0.0f;
        int num = 0;
        while (num < 3 && (iss >> value))
            start_pos[num++] = value;

        for (int i = num; i < 3; ++i)
            start_pos[i] = 0.0f;

        DEBUG_INFO("[Android Slam App Info] start_pos = (%f, %f, %f)", start_pos[0], start_pos[1], start_pos[2]);
    }

}


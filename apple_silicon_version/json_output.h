//
//  json_output.h
//  Simple JSON writer for C++ (no external dependencies)
//

#ifndef JSON_OUTPUT_H
#define JSON_OUTPUT_H

#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <iomanip>

class JSONWriter {
public:
    JSONWriter() : indent_level_(0) {}

    std::string toString() const {
        return stream_.str();
    }

    void startObject() {
        stream_ << "{\n";
        indent_level_++;
    }

    void endObject() {
        indent_level_--;
        stream_ << "\n" << indent() << "}";
    }

    void startArray(const std::string& key) {
        stream_ << indent() << "\"" << key << "\": [\n";
        indent_level_++;
    }

    void endArray() {
        indent_level_--;
        stream_ << "\n" << indent() << "]";
    }

    void addString(const std::string& key, const std::string& value, bool last = false) {
        stream_ << indent() << "\"" << key << "\": \"" << escape(value) << "\"";
        if (!last) stream_ << ",";
        stream_ << "\n";
    }

    void addNumber(const std::string& key, double value, bool last = false) {
        stream_ << indent() << "\"" << key << "\": " << std::fixed << std::setprecision(2) << value;
        if (!last) stream_ << ",";
        stream_ << "\n";
    }

    void addInt(const std::string& key, int value, bool last = false) {
        stream_ << indent() << "\"" << key << "\": " << value;
        if (!last) stream_ << ",";
        stream_ << "\n";
    }

    void addBool(const std::string& key, bool value, bool last = false) {
        stream_ << indent() << "\"" << key << "\": " << (value ? "true" : "false");
        if (!last) stream_ << ",";
        stream_ << "\n";
    }

    void addFloatArray(const std::string& key, const std::vector<float>& values, bool last = false) {
        stream_ << indent() << "\"" << key << "\": [";
        for (size_t i = 0; i < values.size(); ++i) {
            stream_ << std::fixed << std::setprecision(3) << values[i];
            if (i < values.size() - 1) stream_ << ", ";
        }
        stream_ << "]";
        if (!last) stream_ << ",";
        stream_ << "\n";
    }

    void startNestedObject(const std::string& key) {
        stream_ << indent() << "\"" << key << "\": {\n";
        indent_level_++;
    }

    void endNestedObject(bool last = false) {
        indent_level_--;
        stream_ << "\n" << indent() << "}";
        if (!last) stream_ << ",";
        stream_ << "\n";
    }

private:
    std::stringstream stream_;
    int indent_level_;

    std::string indent() const {
        return std::string(indent_level_ * 2, ' ');
    }

    std::string escape(const std::string& str) const {
        std::string result;
        for (char c : str) {
            switch (c) {
                case '\"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default: result += c; break;
            }
        }
        return result;
    }
};

#endif // JSON_OUTPUT_H

import os

class CArtifactGenerator:
    @staticmethod
    def tflite_to_c_array(tflite_path: str, output_c_path: str, array_name: str = "model_tflite"):
        with open(tflite_path, 'rb') as f:
            tflite_content = f.read()
            
        hex_lines = []
        for i in range(0, len(tflite_content), 12):
            chunk = tflite_content[i:i+12]
            hex_chunk = ", ".join([f"0x{byte:02x}" for byte in chunk])
            hex_lines.append(f"    {hex_chunk}")
            
        hex_array_content = ",\n".join(hex_lines)
        
        c_code = f"""// Arquivo gerado pelo Pipeline TinyML (TCC)
#ifndef {array_name.upper()}_H_
#define {array_name.upper()}_H_

#ifdef __has_attribute
#define alignas(x) __attribute__((aligned(x)))
#else
#define alignas(x)
#endif

extern const unsigned char {array_name}[];
extern const int {array_name}_len;

const unsigned char {array_name}[] alignas(16) = {{
{hex_array_content}
}};

const int {array_name}_len = {len(tflite_content)};

#endif // {array_name.upper()}_H_
"""
        with open(output_c_path, 'w') as f:
            f.write(c_code)

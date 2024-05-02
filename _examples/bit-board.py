import numpy as np

# specify the total size of the array
total_size = 3**3
a = np.zeros(total_size + 2, dtype=np.int8)
a[:3] = 1
np.random.shuffle(a)
a[-2:] = 0
print(a)
b = a.copy()
np.random.shuffle(b)
print(b)
c = a.copy()
np.random.shuffle(c)
print(c)

# convert the numpy arrays to bytes
byte_data = [a.tobytes(), b.tobytes(), c.tobytes()]

# specify the file name
file_name = "binary_data.bin"

# write the byte data to the file
with open(file_name, "wb") as f:
    for data in byte_data:
        f.write(data)

print("Data written to file")

arrays = []
with open(file_name, "rb") as f:
    byte_data_read = f.read()
    for i in range(0, len(byte_data_read), total_size + 2):
        arrays.append(
            np.frombuffer(byte_data_read[i : i + total_size + 2], dtype=np.int8)
        )

for array in arrays:
    print(array)

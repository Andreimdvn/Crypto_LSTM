import sys
import os
import subprocess


exe_file = sys.argv[1]
folder = os.getcwd()

ret_count = 0

failed = []

for _, _, file in os.walk(folder):
    if os.path.splitext(file)[1] == '.exe':
        continue
    child = subprocess.Popen('"{}" {}'.format(exe_file, file), stdout=subprocess.PIPE)
    streamdata = child.communicate()[0]
    print(streamdata)
    rc = child.returncode
    if rc == 0:
        ret_count += 1
    else:
        failed.append((file, rc))

print()
print()
print(ret_count)
print(failed)

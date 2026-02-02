## Environment startup for 6050 Assignments


## Start VPN and establish ssh
| STEPS TO LOG IN
| --  | 
| DBLCLK `/Apps/Cisco/Cisco Secure Client.app`; Select extended; use 'push' as pwd |
| DBLCLK Visual Studio Code; CLK lower left corner; Select Host; rivanna |

## Setup environment on rivanna
| STEP |
| -- |
| Open Folder to `/home/dpy8wq` |
| Open terminal with `CTL-~` |
| At terminal `module list miniforge` should show no python |
| Load python with `module load miniforge/24.11.3-py3.12` |
| `module list miniforge` should now show python |
| Run homework the classic way via terminal `python <executable script>` |

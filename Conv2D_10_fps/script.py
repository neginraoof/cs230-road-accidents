#!/usr/bin/python
import os
import platform
import xml.etree.ElementTree as ET
import uuid
import sys,time,subprocess,socket,logging,stat
import threading,subprocess
import asyncore
import pprint, socket, ssl
from datetime import datetime,timedelta

def supported_os_msg():
    print "This script can be run on a machine with below operation systems."
    print "Ubuntu 12.04 and above"
    print "CentOS 6.5 and above"
    print "RHEL 6.7 and above"
    print "Debian 7 and above"
    print "Oracle Linux 6.4 and above"
    print "SLES 12 and above"
    print "OpenSUSE 42.2 and above"

def exitonconfirmation():
    while True:
        ans=raw_input("\nPlease enter 'q/Q' to exit...")
        if 'q' in ans or 'Q' in ans:
            sys.exit()

def log(message):
    logtime = time.strftime("%Y%m%d-%H%M%S")
    logfile.write(("[" + logtime + "] : " + message))
    logfile.write("\n")

def is_supported(OsVersion, LowestVersion):
    return ((OsVersion - LowestVersion >= 0) or (abs(OsVersion - LowestVersion) <= 0.01 ))

def is_package_installated(package):
    proc = subprocess.Popen(["which " + package], stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
    output = proc.stdout.read()
    return True if output else False

def run_shell_cmd(cmd):
    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
    output_line = proc.stdout.read()
    log("Running cmd: " + cmd)
    log(output_line)
    err = proc.stderr.read()
    if err:
        log("Got error")
        log(err)

def get_pkg_installer_cmd_from_os(OsName):
    if OsName in ["Debian", "Ubuntu"]:
        installcmd='apt-get --assume-yes install'
    elif OsName in ["CentOS", "Oracle", "RHEL"]:
        installcmd='yum -y install'
    elif OsName in ["SLES", "OpenSUSE"]:
        installcmd='zypper install'
    else:
        installcmd='apt-get --assume-yes install'
    return installcmd

def install_packages(OsName, packages):
    pkg_installer_cmd = get_pkg_installer_cmd_from_os(OsName)
    for package in packages:
        if package == "iscsiadm":
            if OsName in ["CentOS", "Oracle", "RHEL"]:
                installcmd = pkg_installer_cmd + " iscsi-initiator-utils"
            else:
                installcmd = pkg_installer_cmd + " open-iscsi"
        else:
            installcmd = pkg_installer_cmd + " " + ("acl" if package == "setfacl" else package)
        p = subprocess.Popen([installcmd],shell=True)
        p.wait()
        p = subprocess.Popen(["which " + package],shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output = p.stdout.read()
        err = p.stderr.read()
        log(package +" installation output:"+output+".")
        log(package +" installation Error:"+err)
        if err and not err.isspace():
            log("Failed to install " + package)
            print "Failed to install " + package
            print "Error Details : .%s." % (err)
            exitonconfirmation()

def install_prereq_packages(OsName):
    packages = ["setfacl", "iscsiadm", "lshw"]
    package_map = {}
    for package in packages:
        package_map[package] = is_package_installated(package)
    packages_not_installed = list(filter(lambda package: package_map[package] == False, package_map.keys()))
    if len(packages_not_installed) > 0:
        pkg_msg = ",".join(map(lambda package: "'" + package + "'" ,packages_not_installed))
        if OsName in ["CentOS", "Oracle", "RHEL"]:
            pkg_msg = pkg_msg.replace("iscsiadm", "iscsi-initiator-utils")
        else:
            pkg_msg = pkg_msg.replace("iscsiadm", "open-iscsi")
        print "The script requires " + pkg_msg + " to run"
        print "Do you want us to install " + pkg_msg + " on this machine?"
        ans=raw_input("Please press 'Y' to continue with installation, 'N' to abort the operation. : ")
        if ans in ['y','Y']:
            install_packages(OsName, packages_not_installed)
        elif ans in ['n','N']:
            log("Aborting Installation...")
            print "Aborting Installation..."
            print "Please install " + pkg_msg + " and then run this script again."
            exitonconfirmation()
        else:
            log("You have entered invalid input.:"+ans)
            print "You have entered invalid input. Please try re-running the script."
            exitonconfirmation()

def check_for_open_SSL_TLS_v12():
    p=subprocess.Popen("openssl ciphers -v | awk '{print $2}' | sort | uniq",shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    output = p.stdout.read()
    err = p.stderr.read()
    log("openssl output:"+output+".")
    if err:
        log("openssl Error:"+err)
    if err and not err.isspace() and output.isspace():
        log("error in getting cipher lists or no output")
    elif "TLSv1.2" in output:
        log("TLSv1.2 is supported")
    else:
        log("TLSv1.2 is not supported")
        print "Microsoft Azure File Folder Recovery script needs OpenSSL with TLSv1.2 cipher to securely connect to the recovery point in Azure."
        print "To know whether TLSv1.2 is supported in a OS, run this command."
        print "openssl ciphers -v | awk '{print $2}' | sort | uniq"
        print "The output should show TLSv1.2"
        exitonconfirmation()

## Coremethods of ILR

def CheckForRAIDDisks(LogFolder):
    lshwpath=LogFolder+'/Scripts/lshw2.xml'
    lshwoutput = open(lshwpath,'w')
    proc=subprocess.Popen(['lshw -xml -class disk -class volume'],shell=True,stdout=lshwoutput)
    log('process started')
    proc.wait()
    log('process completed')
    log('process write completed')
    lshwoutput.flush()
    lshwoutput.close()
    tree = ET.parse(lshwpath)
    root = tree.getroot()
    VolumeIndex=1
    raidvolumeslist=list()

    global isStorageSpaceExists
    isStorageSpaceExists = False
    for nodes in root:
        disk= nodes.attrib.get('class')
        if disk == 'disk':
            isMABILRDisk = False
            vendorname="linux"
            disklogicalname=""
            hasvolumes=False
            for child in nodes:
                #print child.tag
                if child.tag == 'vendor':
                    #print child.tag + " " + child.text
                    vendorname = child.text
                    if vendorname == 'MABILR I' :
                        isMABILRDisk = True
                        log('Found MAB ILR Disk')
                        log('Vendor : ' + vendorname)
                elif child.tag == 'logicalname' :
                    disklogicalname=child.text
                    log('Disk Logical Name :' + disklogicalname)
                else :
                    childclass= child.attrib.get('class')
                    #print  isMABILRDisk
                    if childclass == 'volume' :
                        hasvolumes=True
                        description=child.find('description')
                        log("description:"+description.text)
                        logname=child.find('logicalname')
                        if logname != None:
                            log('Logical Volume Name : ' + logname.text)
                            if description.text == "Linux raid autodetect partition" or description.text == "Linux LVM Physical Volume partition":
                                raidvolumeslist.append((disklogicalname+"  |  "+logname.text+"  |  "+description.text))

            log("Has Volumes"+str(hasvolumes))
            if hasvolumes == False :
                log("found disk without volumes")
                diskxmlstring=ET.tostring(nodes)
                if "lvm" in diskxmlstring or "LVM" in diskxmlstring :
                    raidvolumeslist.append((disklogicalname+"  |  -------  |  LVM "))
                    log("Found LVM")

    if len(raidvolumeslist) > 0 :
        isStorageSpaceExists = True
        print "\nPlease find below the logical volume/RAID Array entities present in this machine."
        print "\n************ Volumes from RAID Arrays/LVM partitions ************"
        print "\nSr.No.  |  Disk  |  Volume  |  Partition Type "
        i=1
        for voldetail in raidvolumeslist:
            print "\n"+str(i)+")  | "+voldetail
            i=i+1

def UnMountILRVolumes(LogFolder):

    proc = subprocess.Popen(["mount | grep '"+LogFolder+"'"],
                                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,
                                    )
    err = proc.stderr.read()
    log("Mount List error:"+err)
    processoutput=proc.stdout.readlines()
    log("Mount List Output Len :%d" % (len(processoutput)))
    log("Mount List Output :%s." % (processoutput))
    if len(processoutput) > 0:
        for record in processoutput:
            log("UnMount Record :"+record)
            values = record.split(' ')
            log("UnMount Record :"+values[0])
            proc = subprocess.Popen(["umount '"+values[0]+"'"],
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,
                                )
            output = proc.stdout.read()
            log("UnMount ouput:"+output)
            err = proc.stderr.read()
            log("UnMount error:"+err)

def logout_targets(target_node_addr):
    proc = subprocess.Popen(["iscsiadm -m session"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
    output_lines = proc.stdout.readlines()
    err_lines = proc.stderr.readlines()
    log("Output:" + str(output_lines))
    log("Error:" + str(err_lines))
    for line in output_lines:
        if target_node_addr in line:
            addr = line.split()
            iscsci_target_addr = filter(lambda x:x.startswith('iqn.2016-01.microsoft.azure.backup'), addr)
            if len(iscsci_target_addr) == 1:
                target_addr = iscsci_target_addr[0].strip()
                proc = subprocess.Popen(["iscsiadm -m node -T "+target_addr+" --logout"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
                output = proc.stdout.read()
                err = proc.stderr.read()
                log("Logging out target: "+ target_addr)
                log("Output:" + output)
                log("Error:" + err)
                if "successful." in output:
                    log("Logout Succeeded for target:"+ target_addr)

def MountILRVolumes(LogFolder):
    time.sleep(5)
    lshwpath = LogFolder+'/Scripts/lshw1.xml'
    lshwoutput = open(lshwpath,'w')
    proc=subprocess.Popen(['lshw -xml -class disk -class volume'],shell=True,stdout=lshwoutput)
    log('process started')
    proc.wait()
    log('process completed')
    log('process write completed')
    lshwoutput.flush()
    lshwoutput.close()
    tree = ET.parse(lshwpath)
    root = tree.getroot()
    VolumeIndex = 1
    volumeslist = list()
    raidvolumeslist = list()
    failedvolumeslist = list()
    for nodes in root:
        disk= nodes.attrib.get('class')
        if disk == 'disk':
            isMABILRDisk = False
            vendorname = "linux"
            disklogicalname = ""
            hasvolumes = False
            for child in nodes:
                #print child.tag
                if child.tag == 'vendor':
                    #print child.tag + " " + child.text
                    vendorname = child.text
                    if vendorname == 'MABILR I':
                        isMABILRDisk = True
                        log('Found MAB ILR Disk')
                        log('Vendor : ' + vendorname)
                elif child.tag == 'logicalname' and isMABILRDisk :
                    disklogicalname=child.text
                    log('Disk Logical Name :' + disklogicalname)
                else :
                    childclass = child.attrib.get('class')
                    #print  isMABILRDisk
                    if childclass == 'volume' and isMABILRDisk :
                        hasvolumes=True
                        description=child.find('description')
                        log("description:"+description.text)
                        logname=child.find('logicalname')
                        if logname != None:
                            log('Logical Volume Name : ' + logname.text)
                            if description.text == "Linux raid autodetect partition" or description.text == "Linux LVM Physical Volume partition":
                                raidvolumeslist.append((disklogicalname+"  |  "+logname.text+"  |  "+description.text))
                            else:
                                MountPath = LogFolder+'/Volume'+str(VolumeIndex)
                                VolumeIndex = VolumeIndex+1
                                log(MountPath)
                                os.mkdir(MountPath)
                                log("Mounting volume"+(logname.text)+" to path "+(MountPath))
                                proc = subprocess.Popen(["mount",logname.text,MountPath],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                                output = proc.stdout.read()
                                err = proc.stderr.read()
                                log("Mount Output:"+output+".")
                                log("Mount Error:"+err)
                                if err and not err.isspace():
                                    log("Mount failed for volume"+(logname.text)+" to path "+(MountPath))
                                    log("Retry: Mounting with nouuid option for volume"+(logname.text)+" to path "+(MountPath))
                                    proc = subprocess.Popen(["mount","-o","nouuid",logname.text,MountPath],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                                    output = proc.stdout.read()
                                    err = proc.stderr.read()
                                    log("Mount Output:"+output+".")
                                    log("Mount Error:"+err)
                                    if err and not err.isspace():
                                        log("Retry mount failed for volume"+(logname.text)+" to path "+(MountPath))
                                        failedvolumeslist.append((disklogicalname+"  |  "+logname.text+"  |  "+description.text))
                                    else:
                                        volumeslist.append((disklogicalname+"  |  "+logname.text+"  |  "+MountPath))
                                else:
                                    volumeslist.append((disklogicalname+"  |  "+logname.text+"  |  "+MountPath))
            log("Has Volumes"+str(hasvolumes))
            if hasvolumes == False and isMABILRDisk :
                log("found MAB ILR disk without volumes")
                diskxmlstring=ET.tostring(nodes)
                if "lvm" in diskxmlstring or "LVM" in diskxmlstring :
                    raidvolumeslist.append((disklogicalname+"  |  -------  |  LVM "))
                    log("Found LVM")
                else:
                    fschk_cmd = "lsblk -f " + disklogicalname.strip()
                    fsp=subprocess.Popen(fschk_cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
                    fsp.wait()
                    output = fsp.stdout.readlines()
                    err = fsp.stderr.read()
                    is_error = err or (len(output) != 2)
                    if is_error:
                        log("Got error while checking filesystem on disks without volumes")
                        log(err)
                        log(str(output))
                    else:
                        file_system_output = output[1].strip().split()
                        if len(file_system_output) > 1 and file_system_output[1]:
                            print "\nIdentified the below disk which does not have volumes."
                            print "\n " + disklogicalname
                            ans=raw_input("Please press 'Y' to continue with mouting this disk without volume, 'N' to abort the operation. : ")
                            if ans in ['y','Y']:
                                MountPath=LogFolder+'/Disk'+str(VolumeIndex)
                                VolumeIndex=VolumeIndex+1
                                log(MountPath)
                                os.mkdir(MountPath)
                                log("Mounting disk"+(disklogicalname)+" to path "+(MountPath))
                                proc=subprocess.Popen("mount " + disklogicalname + " " +MountPath,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
                                output = proc.stdout.read()
                                error = proc.stderr.read()
                                log("Mount Output for Disk without volumes:"+output+".")
                                log("Mount Error for Disk without volumes:"+err)
                                if error:
                                    log("Got error while mounting disk without volumes")
                                    log(error)
                                    log("Mount failed for disk"+(disklogicalname)+" to path "+(MountPath))
                                    log("Retry: Mounting with nouuid option for disk"+(disklogicalname)+" to path "+(MountPath))
                                    proc = subprocess.Popen("mount -o nouuid " + disklogicalname + " " + MountPath,stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
                                    output = proc.stdout.read()
                                    err = proc.stderr.read()
                                    log("Mount Output:"+output+".")
                                    log("Mount Error:"+err)
                                    if err and not err.isspace():
                                        log("Retry mount failed for volume"+(disklogicalname)+" to path "+(MountPath))
                                        failedvolumeslist.append((disklogicalname+"  |             |  "+description.text))
                                    else:
                                        volumeslist.append((disklogicalname+"  |             |  "+MountPath))
                                else:
                                    log("Mounted disklogicalname")
                                    log(disklogicalname)
                                    volumeslist.append(disklogicalname + "  |             |  " + MountPath)

    if len(volumeslist) > 0:
        print "\n************ Volumes of the recovery point and their mount paths on this machine ************"
        print "\nSr.No.  |  Disk  |  Volume  |  MountPath "
        i = 1
        for voldetail in volumeslist:
            print "\n"+str(i)+")  | "+voldetail
            i = i+1
    else:
        print "\n0 volumes mounted as volumes are either RAID Arrays/LVM partitions or failed to mount."

    if len(raidvolumeslist) > 0:
        print "\n************ Volumes from RAID Arrays/LVM partitions ************"
        print "\nSr.No.  |  Disk  |  Volume  |  Partition Type "
        i = 1
        for voldetail in raidvolumeslist:
            print "\n"+str(i)+")  | "+voldetail
            i=i+1
        print "\nRun the following commands to mount and bring the partitions online."
        print "\nFor LVM partitions:"
        print "\n    $ pvs <volume name as shown above> - To list the volume group names under this physical volume"
        print "\n    $ lvdisplay <volume-group-name from the above command's result> - To list all logical volumes, names and their paths in this volume group"
        print "\n    $ mount <LV path> </mountpath> - To mount the logical volumes to the path of your choice"
        print "\nFor RAID Arrays:"
        print "\n    $ mdadm --detail --scan (To display details about all raid disks)"
        print "    The relevant RAID disk will be named as '/dev/mdm/<RAID array name in the backed up VM>'"
        print "\n    Use the mount command if the RAID disk has physical volumes"
        print "    $ mount <RAID Disk Path> </mountpath>"
        print "\n    If this RAID disk has another LVM configured in it then follow the same prcedure as outlined above for LVM partitions with the volume name being the RAID Disk name"
    if len(failedvolumeslist) > 0:
        print "\nThe following partitions failed to mount since the OS couldn't identify the filesystem."
        print "\n************ Volumes from unknown filesystem ************"
        print "\nSr.No.  |  Disk  |  Volume  |  Partition Type "
        i = 1
        for voldetail in failedvolumeslist:
            print "\n"+str(i)+")  | "+voldetail
            i = i+1
        print "\nPlease refer to '"+LogFolder+ "/Scripts/MicrosoftAzureBackupILRLogFile.log' for more details."
    print "\n************ Open File Explorer to browse for files. ************"

def UpdateISCSIConfig(logfolder,TargetUserName,TargetPassword):
    iscsi_config_file='/etc/iscsi/iscsid.conf'
    iscsi_config_temp_file1=logfolder+"/Scripts/iscsidtemp1.conf"
    iscsi_config_temp_file2=logfolder+"/Scripts/iscsidtemp2.conf"
    iscsiconfig=open(iscsi_config_temp_file1,'w+')
    iscsiconfig.write("discovery.sendtargets.auth.authmethod =\n")
    iscsiconfig.write("discovery.sendtargets.auth.authmethod=\n")
    iscsiconfig.write("discovery.sendtargets.auth.authmethod  \n")
    iscsiconfig.write("discovery.sendtargets.auth.username =\n")
    iscsiconfig.write("discovery.sendtargets.auth.username=\n")
    iscsiconfig.write("discovery.sendtargets.auth.username  \n")
    iscsiconfig.write("discovery.sendtargets.auth.password =\n")
    iscsiconfig.write("discovery.sendtargets.auth.password=\n")
    iscsiconfig.write("discovery.sendtargets.auth.password  \n")
    iscsiconfig.write("discovery.sendtargets.auth.username_in =\n")
    iscsiconfig.write("discovery.sendtargets.auth.username_in=\n")
    iscsiconfig.write("discovery.sendtargets.auth.username_in  \n")
    iscsiconfig.write("discovery.sendtargets.auth.password_in =\n")
    iscsiconfig.write("discovery.sendtargets.auth.password_in=\n")
    iscsiconfig.write("discovery.sendtargets.auth.password_in  \n")
    iscsiconfig.write("node.session.auth.authmethod =\n")
    iscsiconfig.write("node.session.auth.authmethod=\n")
    iscsiconfig.write("node.session.auth.authmethod  \n")
    iscsiconfig.write("node.session.auth.username =\n")
    iscsiconfig.write("node.session.auth.username=\n")
    iscsiconfig.write("node.session.auth.username  \n")
    iscsiconfig.write("node.session.auth.password =\n")
    iscsiconfig.write("node.session.auth.password=\n")
    iscsiconfig.write("node.session.auth.password  \n")
    iscsiconfig.write("node.session.auth.username_in =\n")
    iscsiconfig.write("node.session.auth.username_in=\n")
    iscsiconfig.write("node.session.auth.username_in  \n")
    iscsiconfig.write("node.session.auth.password_in =\n")
    iscsiconfig.write("node.session.auth.password_in=\n")
    iscsiconfig.write("node.session.auth.password_in  \n")
    iscsiconfig.close()
    log("Removing old iscsi config entries.")
    updatediscsiconfig=open(iscsi_config_temp_file2,'w+')
    p=subprocess.Popen(["grep -v -f "+iscsi_config_temp_file1+" "+iscsi_config_file],
                            stdout=updatediscsiconfig,shell=True,
                            )
    p.wait()

    log("Removed old iscsi config entries.")
    updatediscsiconfig.write("\ndiscovery.sendtargets.auth.authmethod = CHAP")
    updatediscsiconfig.write("\ndiscovery.sendtargets.auth.username = "+OSName+TargetUserName)
    updatediscsiconfig.write("\ndiscovery.sendtargets.auth.password = "+TargetPassword)
    updatediscsiconfig.write("\n#discovery.sendtargets.auth.username_in = username_in")
    updatediscsiconfig.write("\n#discovery.sendtargets.auth.password_in = password_in")
    updatediscsiconfig.write("\nnode.session.auth.authmethod = CHAP")
    updatediscsiconfig.write("\nnode.session.auth.username = "+OSName+TargetUserName)
    updatediscsiconfig.write("\nnode.session.auth.password = "+TargetPassword)
    updatediscsiconfig.write("\n#node.session.auth.username_in = username_in")
    updatediscsiconfig.write("\n#node.session.auth.password_in = password_in\n")
    log("successfully added new iscsi config entries.")
    updatediscsiconfig.flush()
    updatediscsiconfig.close()
    p=subprocess.Popen(["cp "+iscsi_config_temp_file2+" "+iscsi_config_file],
                            stdout=subprocess.PIPE,shell=True,
                            )
    output = p.stdout.read()
    log("CP output:"+output)

    log("/etc/iscsi/iscsid.conf file is replaced successfully.")

def discovery_error_prompt(params):
    err = params['error']
    LogFolder = params['LogFolder']
    TargetNodeAddress = params['TargetNodeAddress']
    TargetPortalAddress = params['TargetPortalAddress']
    TargetPortalPortNumber = params['TargetPortalPortNumber']
    if "initiator failed authorization" in err:
        log("Discovery Failed.")
        print "\nThis script cannot connect to the recovery point. Either the password entered is invalid or the disks have been unmounted."
        print "Please enter the correct password or download a new script from the portal."
    elif "iscsid is not running" in err:
        log("Discovery Failed.")
        log("Failure Reason: iscsid is not running")
        print "\nException caught while connecting to the recovery point."
        print "\nFailure Reason: iscsid is not running. can not connect to iSCSI daemon (111)."
        print "\nPlease refer to the logs at '"+ LogFolder +"/Scripts'. You can also retry running the script from another machine. If problem persists, raise a support request with details about OS of machines where script was run and the entire log folder"
    else:
        log("Discovery Failed.")
        log("Target Not Found :"+TargetNodeAddress)
        log("Unable to acces the target URL : "+TargetPortalAddress+":"+str(TargetPortalPortNumber))
        log("Use below curl command to check the access to any URL and Port")
        log("curl "+TargetPortalAddress+":"+str(TargetPortalPortNumber)+" --connect-timeout 2")
        log("It will display this message if you have access. 'curl: (56) Failure when receiving data from the peer'")
        log("Else it will get timed out with message 'curl: (28) connect() timed out!'")
        print "\nUnable to access Recovery vault, check your proxy/firewall setting to ensure access to <"+TargetPortalAddress+":"+str(TargetPortalPortNumber)+">."
        print "\nIn general, make sure you meet the network connectivity requirements to Azure Recovery vault as specified here: https://docs.microsoft.com/en-us/azure/backup/backup-azure-vms-prepare#network-connectivity"
        print "\nIf problem persists despite meeting all the network connectivity requirements as specified above, please refer to the logs at '"+ LogFolder +"/Scripts'"


def ILRMain(ilr_params):
    LogFolder = ilr_params['LogFolder']
    ScriptId = ilr_params['ScriptId']
    MinPort = ilr_params['MinPort']
    MaxPort = ilr_params['MaxPort']
    TargetPortalAddress = ilr_params['TargetPortalAddress']
    TargetPortalPortNumber = ilr_params['TargetPortalPortNumber']
    TargetNodeAddress = ilr_params['TargetNodeAddress']
    TargetUserName = ilr_params['TargetUserName']
    TargetPassword = ilr_params['TargetPassword']
    VMName = ilr_params['VMName']
    MachineName = ilr_params['MachineName']
    docleanup = ilr_params['DoCleanUp']
    global OSName
    OSName = ilr_params['OsNameVersion']
    IsMultiTarget = ilr_params['IsMultiTarget']
    LogFileName = LogFolder + "/Scripts/MicrosoftAzureBackupILRLogFile.log"

    log("Log Folder Path: " + LogFolder)
    log("Log File Path: " + LogFileName)
    log("Script Id: " + ScriptId)
    log("MinPort: " +  str(MinPort))
    log("MaxPort: " +  str(MaxPort))
    log("TargetPortalAddress: " + TargetPortalAddress)
    log("TargetPortalPortNumber: " + str(TargetPortalPortNumber))
    log("TargetNodeAddress: " + TargetNodeAddress)
    log("IsMultiTarget: " + IsMultiTarget)


    if docleanup:
        log("Only Cleanup called")
        print "\nRemoving the local mount paths of the currently connected recovery point..."

    if TargetPassword == "UserInput012345" and not docleanup :
        TargetPassword=raw_input("Please enter the password as shown on the portal to securely connect to the recovery point. : ")
        log("Input Password Length: "+str(len(TargetPassword)))
        if len(TargetPassword) != 15 :
            log("Password length is not 15 char. ")
            print "\nYou need to enter the complete 15 character password as shown on the portal screen. Please use the copy button beside the generated password and past here."
            exitonconfirmation()

    scriptran=False
    isProcessRunning=False
    proc = subprocess.Popen(["tail","-n","1","/etc/MicrosoftAzureBackupILR/mabilr.conf"],
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                        )
    processoutput=proc.stdout.readlines()
    log("Output Len :%d" % (len(processoutput)))
    log("Output :%s." % (processoutput))
    if len(processoutput) > 0:
        lastrecord = processoutput[0].split('\n')
        log("Last Record in MABILR Config :%s." % (lastrecord))
        values = lastrecord[0].split(',')
        portnumber=values[4]
        processid=values[5]
        targetnodeaddress=values[3]
        lastvmname=values[6]
        lastlogfolder=values[7]
        scriptran=True
        proc=subprocess.Popen(["iscsiadm -m session"],
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,
                        )
        output = proc.stdout.read().lower()
        err = proc.stderr.read()
        log("Session Target Output:"+output+".")
        log("Session Target Error:"+err)
        target_address_prefix = targetnodeaddress if IsMultiTarget == "0" else targetnodeaddress[:targetnodeaddress.rfind('.')].lower()
        if target_address_prefix in output:
            if docleanup :
                ans='Y'
            else:
                print "\nWe detected a session already connected to a recovery point of the VM '"+lastvmname+"'."
                print "We need to unmount the volumes before connecting to the new recovery point of '"+VMName+"'"
                ans=raw_input("\nPlease enter 'Y' to proceed or 'N' to abort...")

            if 'y' in ans or 'Y' in ans:
                log("Un mounting existing mount points.")
                UnMountILRVolumes(lastlogfolder)
                logout_targets(target_address_prefix)
                if not docleanup:
                    print "\nOlder session disconnected. Establishing a new session for the new recovery point...."
            else:
                print "It is recommended to close the earlier session before starting new connection to another RP."
                exitonconfirmation()
        else:
            UnMountILRVolumes(lastlogfolder)

    if not docleanup:
        hostname=socket.gethostname()
        log("Host Name :"+hostname)
        vmname = VMName.split(';')
        if (hostname.lower() == vmname[2].lower() or hostname.lower() == MachineName.lower()):
            CheckForRAIDDisks(LogFolder)
            log("isStorageSpaceExists"+str(isStorageSpaceExists))
            if isStorageSpaceExists == True :
                print "\nMount the recovery point only if you are SURE THAT THESE ARE NOT BACKED UP/ PRESENT IN THE RECOVERY POINT."
                print "If they are already present, it might corrupt the data irrevocably on this machine."
                print "It is recommended to run this script on any other machine with similar OS to recovery files."
                ans=raw_input("\nShould the recovery point be mounted on this machine? ('Y'/'N') ")
                if 'y' in ans or 'Y' in ans:
                    log("user selected to continue")
                else:
                    print "\nPlease run this script on any other machine with similar OS to recover files."
                    exitonconfirmation()
        UpdateISCSIConfig(LogFolder,TargetUserName,TargetPassword)

    if scriptran == True:
        log("Script already ran earlier on this machine.")
        log("PortNumber:%s, PID:%s" % (portnumber,processid))

        proc = subprocess.Popen(["ps","-p",processid,"-o","comm="],
                        stdout=subprocess.PIPE,
                        )
        processoutput=proc.stdout.readlines()
        #print "Output Len :%d" % (len(processoutput))
        #print "Output :%s." % (processoutput)
        if len(processoutput) > 0:
            processname = processoutput[0].split('\n')
            log("Process Name :%s." % (processname[0]))
            if processname[0] == "SecureTCPTunnel":
                log("SecureTCPTunnel process is already running")
                isProcessRunning=False
                log("Killing the existing SecureTCPTunnelProcess to free 3260 port in OSName:"+OSName)
                proc = subprocess.Popen(["kill","-9",processid],
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE
                        )
                output = proc.stdout.read()
                err = proc.stderr.read()
                log("Kill output:"+output)
                log("Kill error:"+err)
            else:
                log("SecureTCPTunnel process is not running")
    else:
        log("Script didnt' ran earlier on this machine.")

    if docleanup:
        log("Cleanup Completed")
        print "\nThe local mount paths have been removed."
        print "\nPlease make sure to click the 'Unmount disks' from the portal to remove the connection to the recovery point."
        exitonconfirmation()

    if isProcessRunning == False:
        try:
            if "CentOS" in OSName or "Oracle" in OSName or "RHEL" in OSName:
                MinPort=3260
                MaxPort=3260
            log("Starting SecureTCPTunnel process...")
            log("with args:")
            log(LogFolder + "/Scripts/SecureTCPTunnel.py " +  OSName + " " + LogFolder + " " + ScriptId + " " + str(MinPort) + " " + str(MaxPort) + " " + TargetPortalAddress + " " + str(TargetPortalPortNumber) + " " + TargetNodeAddress + " " + VMName)
            proc = subprocess.Popen([LogFolder + "/Scripts/SecureTCPTunnel.py",OSName,LogFolder,ScriptId,str(MinPort),str(MaxPort),TargetPortalAddress,str(TargetPortalPortNumber),TargetNodeAddress,VMName],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE
                )
            pid=proc.pid
            log("pid : " + str(pid))
        except Exception as e:
            log("Exception raised while starting SecureTCPTunnel process")
            log(repr(e))
            if proc.stdout:
                output = proc.stdout.read()
                log("SecureTCPTunnel output:"+output)
            if proc.stderr:
                err = proc.stderr.read()
                log("SecureTCPTunnel error:"+err)
        found=True
        maxretrycount=2
        retrycount=0
        while found and retrycount < maxretrycount:
            try:
                time.sleep(1)

                with open('/etc/MicrosoftAzureBackupILR/mabilr.conf','r') as f:
                        lines = f.readlines()
                        for line in lines:
                            log(line)
                            values = line.split(',')
                            if values[0] == ScriptId and values[5] == str(pid):
                                log("Secure TCP process added record to config file")
                                found=False
                                portnumber=values[4]
            except Exception as e:
                log("Exception raised while reading mabilr.conf file")
                #log("Exception: "+e.message)
            retrycount=retrycount+1
        if retrycount == maxretrycount:
            log("Secure TCP Process is not started after max retry count")
            if "CentOS" in OSName or "Oracle" in OSName or "RHEL" in OSName:
                print "We are unable to communicate via the port 3260 on this machine since it is being used by ISCSI target server or any other application. Please unblock the port or use another machine where the port is open for communicatoin."
            else:
                print "We are unable to use local port range "+str(MinPort)+"-"+str(MaxPort)+" for our communication on this machine. Please check if these ports are already being used by another application."
            print "Please refer to the logs at '"+ LogFolder +"/Scripts'"
            exitonconfirmation()
        else:
            log("SecureTCPTunnel started on  Port %s" % (portnumber))

    print "\nConnecting to recovery point using ISCSI service..."
    log(("Discovering Targets from Portal: "+TargetPortalAddress+","+str(TargetPortalPortNumber)))
    p=subprocess.Popen(["iscsiadm -m discovery -t sendtargets -p 127.0.0.1:"+portnumber],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,
                            )
    if IsMultiTarget == "0":
        output = p.stdout.read()
        err = p.stderr.read()
        log("Discovery Output:"+output+".")
        log("Discovery Error:"+err)

        is_not_ready = "notready" in output
        num_retries_left = 4
        while num_retries_left > 0 and is_not_ready == True:
            print "This is a large disk. The target is not ready yet. Waiting for 5 mins and trying again."
            log("This is a large disk. The target is not ready yet. Waiting for 5 mins and trying again.")
            time.sleep(300)
            p=subprocess.Popen(["iscsiadm -m discovery -t sendtargets -p 127.0.0.1:"+portnumber],
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,
                                )
            output = p.stdout.read()
            err = p.stderr.read()
            log("Discovery Output:"+output+".")
            log("Discovery Error:"+err)
            is_not_ready = "notready" in output
            num_retries_left = num_retries_left - 1
        if is_not_ready == True:
            print "The target is not ready yet for large disk. Please retry after 10 mins."
            log("The target is not ready yet for large disk. Please retry after 10 mins.")
            exit()

        if ("127.0.0.1:"+str(portnumber)+",-1 "+TargetNodeAddress) in output:
            log("Discovery Succeeded.")
            log("Target Found: "+TargetNodeAddress)
            log("Connecting to target "+TargetNodeAddress+" ...")
            connection_params = {
                    "TargetNodeAddress" : TargetNodeAddress,
                    "LogFolder" : LogFolder,
                    "LocalPortNumber" : portnumber,
                    "IsMultiTarget" : IsMultiTarget
            }
            connection_status = connect_to_target(connection_params)
            if connection_status == False:
                discovery_params = {
                    "error" : err,
                    "LogFolder" : LogFolder,
                    "TargetNodeAddress" : TargetNodeAddress,
                    "TargetPortalAddress" : TargetPortalAddress,
                    "TargetPortalPortNumber" : TargetPortalPortNumber
                }
                discovery_error_prompt(discovery_params)
    else:
        output_lines = p.stdout.readlines()
        err = p.stderr.readlines()
        log("Discovery Output:"+str(output_lines)+".")
        log("Discovery Error:"+str(err))
        not_ready_list = filter(lambda x: "notready" in x.lower(), output_lines)
        num_retries_left = 4
        while num_retries_left > 0 and len(not_ready_list) >= 1:
            print "This is a large disk. The target is not ready yet. Waiting for 5 mins and trying again."
            log("This is a large disk. The target is not ready yet. Waiting for 5 mins and trying again.")
            time.sleep(300)
            p=subprocess.Popen(["iscsiadm -m discovery -t sendtargets -p 127.0.0.1:"+portnumber],
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,
                                )
            output_lines = p.stdout.readlines()
            err = p.stderr.readlines()
            log("Discovery Output:"+str(output_lines)+".")
            log("Discovery Error:"+str(err))
            not_ready_list = filter(lambda x: "notready" in x.lower(), output_lines)
            num_retries_left = num_retries_left - 1
        if len(not_ready_list) >= 1:
            print "The target is not ready yet for large disk. Please retry after 10 mins."
            log("The target is not ready yet for large disk. Please retry after 10 mins.")
            exit()
        target_addresses = list()
        is_discovery_success = False
        address_separator_index = TargetNodeAddress.rfind('.')
        target_prefix = TargetNodeAddress[:address_separator_index-1]
        target_sequence_num = TargetNodeAddress[address_separator_index+1:]
        for output in output_lines:
            if ("127.0.0.1:"+str(portnumber)) in output:
                log("Discovery Succeeded.")
                (iscsi_local_addr, iscsi_params) = output.split(",")
                iscsi_params = iscsi_params.strip()
                (disknum, iscsi_target_node_address) = iscsi_params.split(' ')
                if target_prefix in iscsi_target_node_address and target_sequence_num in iscsi_target_node_address and "notready" not in iscsi_target_node_address:
                    log("Target Found: "+ iscsi_target_node_address + " target num:" + disknum)
                    log("Appending the target to the list "+iscsi_target_node_address+" ...")
                    is_discovery_success = True
                    target_addresses.append(iscsi_target_node_address)
        if is_discovery_success == False:
            discovery_params = {
                "error" : err,
                "LogFolder" : LogFolder,
                "TargetNodeAddress" : TargetNodeAddress,
                "TargetPortalAddress" : TargetPortalAddress,
                "TargetPortalPortNumber" : TargetPortalPortNumber
            }
            discovery_error_prompt(discovery_params)
        connection_params = {
            "TargetNodeAddress" : target_addresses,
            "LogFolder" : LogFolder,
            "LocalPortNumber" : portnumber,
            "IsMultiTarget" : IsMultiTarget
        }
        connect_to_target(connection_params)
    exitonconfirmation()

def connect_to_target(connection_params):
    portnumber = connection_params['LocalPortNumber']
    LogFolder = connection_params['LogFolder']
    TargetNodeAddress = connection_params['TargetNodeAddress']
    IsMultiTarget = connection_params['IsMultiTarget']
    connection_status = True
    output = ""
    err = ""
    if IsMultiTarget == "1":
        for target_node_addr in TargetNodeAddress:
            p=subprocess.Popen(["iscsiadm -m node -T "+target_node_addr+" -p 127.0.0.1:"+portnumber+" --login"],
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,
                                )
            output = p.stdout.read()
            err = p.stderr.read()
            log("Connect Target Output:"+output+".")
            log("Connect Target Error:"+err)
            connection_status = connection_status and (("successful." in output) or ("iscsiadm: default: 1 session requested, but 1 already present." in err or (not (output and not output.isspace()))))
    else:
        p=subprocess.Popen(["iscsiadm -m node -T "+TargetNodeAddress+" -p 127.0.0.1:"+portnumber+" --login"],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,
                            )
        output = p.stdout.read()
        err = p.stderr.read()
        log("Connect Target Output:"+output+".")
        log("Connect Target Error:"+err)
        connection_status = (("successful." in output) or ("iscsiadm: default: 1 session requested, but 1 already present." in err or (not (output and not output.isspace()))))
    if connection_status == True:
        if "successful." in output:
            log("Connection Succeeded.")
            print "\nConnection succeeded!"
            print "\nPlease wait while we attach volumes of the recovery point to this machine..."
            log("Mounting Volumes to the Mount Paths.")
            MountILRVolumes(LogFolder)
            log("Mounting of volumes completed successfully.")
            print "\nAfter recovery, remove the disks and close the connection to the recovery point by clicking the 'Unmount Disks' button from the portal or by using the relevant unmount command in case of powershell or CLI."
            print "\nAfter unmounting disks, run the script with the parameter 'clean' to remove the mount paths of the recovery point from this machine."
            connection_status = True
        elif "iscsiadm: default: 1 session requested, but 1 already present." in err or (not (output and not output.isspace())):
            log("Already connected to target.")
            print "\nThe target has already been logged in via an iSCSI session."
            log("Mounting Volumes to the Mount Paths.")
            MountILRVolumes(LogFolder)
            log("Mounting of volumes completed successfully.")
            print "\nAfter recovery, remove the disks and close the connection to the recovery point by clicking the 'Unmount Disks' button from the portal or by using the relevant unmount command in case of powershell or CLI."
            print "\nAfter unmounting disks, run the script with the parameter 'clean' to remove the mount paths of the recovery point from this machine."
            connection_status = True
        else:
            connection_status = False
    return connection_status

def generate_securetcptunnel_code(script_folder):
    SecureTCPTunnelCode ="""#!/usr/bin/python
import threading,subprocess
import time
import asyncore
import pprint, socket, ssl
import logging,sys,os
from datetime import datetime,timedelta
class SecureTCPTunnelServer(asyncore.dispatcher):

    def __init__(self, port_range, ILRTargetInfo, ilr_config_file):

        self.logger = logging.getLogger('SecureTCPTunnelServer')

        asyncore.dispatcher.__init__(self)
        ilrconfig = open(ilr_config_file,"a+")
        minport, maxport = port_range
        port = minport
        self.ILRTargetInfo = ILRTargetInfo
        TargetPortalAddress, TargetPortalPortNumber,TargetNodeAddress,ScriptId,VMName,LogFolder = ILRTargetInfo
        gcthread = GCThread("GCThread",TargetNodeAddress)
        gcthread.start()
        while port <= maxport:
            try:
                SocketAddress = ('localhost', port)
                self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
                self.bind(SocketAddress)
                self.address = self.socket.getsockname()
                self.listen(1)
                self.logger.info('Listening on %s', self.address)
                ilrconfig.write("\\n%s,%s,%d,%s,%d,%d,%s,%s" % (ScriptId,TargetPortalAddress,TargetPortalPortNumber,TargetNodeAddress,port,os.getpid(),VMName,LogFolder))
                ilrconfig.close()
                break
            except socket.error, (value,message):
                self.logger.error("socket.error - %d, Port: %d - %s" % (value,port,message))
                if value == 98:
                    self.close()
                    port=port+1
                else:
                    break
        return

    def handle_accept(self):

        client_info = self.accept()

        self.logger.info('Accepted client connection from %s', client_info[1])

        try:
            cthread=ClientThread("ClientThread",clientsock=client_info[0],ILRTargetInfo=self.ILRTargetInfo)
            cthread.start()
        except Exception as e:
            self.logger.warning("Exception raised while creating client thread")
            #self.logger.warning("Exception: "+e.message)
        return


    def handle_close(self):

        self.logger.info('Closing the Server.')

        self.close()

        return

class GCThread (threading.Thread):

    def __init__(self, name, targetNodeAddress):
        self.logger = logging.getLogger('GCThread')
        threading.Thread.__init__(self)
        self.name = name
        self.TargetNodeAddress = targetNodeAddress

    def run(self):
        self.logger.info("Starting " + self.name)
        self.endtime = datetime.now()+timedelta(minutes=720)
        self.logger.info("GC end Time " + str(self.endtime.strftime("%Y%m%d%H%M%S")))
        while True:
            time.sleep(60)
            self.logger.info("GC Started")
            self.logger.info("GC current Time " + str(datetime.now().strftime("%Y%m%d%H%M%S")))
            if self.endtime < datetime.now():
                self.logger.info("SecureTCPTunnel reached 12 hours active window. Killing the process.")
                try:
                    p = subprocess.Popen(["iscsiadm -m node -T "+TargetNodeAddress+" --logout"],
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,
                                )
                    output = p.stdout.read()
                    err = p.stderr.read()
                    self.logger.info("Logout Target Output:"+output+".")
                    self.logger.info("Logout Target Error:"+err)
                    if "successful." in output:
                        self.logger.info("Logout Succeeded.")
                except Exception as e:
                    self.logger.warning("Exception raised while creating client thread")
                    #self.logger.warning("Exception: "+e.message)
                self.logger.info("GC Completed")
                os._exit(0)


class ServerThread (threading.Thread):

    def __init__(self, name, clientsock, serversock):
        self.logger = logging.getLogger('ServerThread')
        threading.Thread.__init__(self)
        self.name = name
        self.chunk_size = 131072
        self.clientsocket = clientsock
        self.serversocket = serversock

    def run(self):

        self.logger.info("Starting " + self.name)
        try:
            while True:
                self.logger.info("reading from server")
                data = self.serversocket.recv(self.chunk_size)
                if data != '':
                    self.logger.info("sending to client")
                    sent=self.clientsocket.send(data[:len(data)])
                    self.logger.info("sent to client (%d)" % (sent) )
                elif data == '':
                    break

        except socket.error, (value,message):
            self.logger.error('socket.error - ' + message)

        self.logger.info("Disconnected from Server")
        self.clientsocket.close()
        self.serversocket.close()

class ClientThread (threading.Thread):

    def __init__(self, name, clientsock, ILRTargetInfo):
        self.logger = logging.getLogger('ClientThread')
        threading.Thread.__init__(self)
        self.logger.info("Creating Client Thread for new connection")
        self.name = name
        self.chunk_size=131072
        self.clientsocket=clientsock
        TargetPortalAddress, TargetPortalPortNumber,TargetNodeAddress,ScriptId,VMName,LogFolder = ILRTargetInfo
        self.logger.info("LogFolder:"+LogFolder)
        #self.logger.info("System Version: %s" % (sys.version_info))
        if sys.version_info < (2,7,9):
            ssocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            self.serversocket = ssl.wrap_socket(ssocket)

            self.serversocket.connect((TargetPortalAddress, TargetPortalPortNumber))
            #print "connection succeeded %s %s" % (TargetPortalAddress, TargetPortalPortNumber)

            #cert = self.serversocket.getpeercert()
            #print repr(self.serversocket.getpeername())
            #print pprint.pformat(self.serversocket.getpeercert())
            #print self.serversocket.cipher()
        else:
            context = ssl.create_default_context()
            context = ssl.SSLContext(ssl. PROTOCOL_TLSv1_2)
            context.verify_mode = ssl.CERT_OPTIONAL
            context.check_hostname = True
            context.load_default_certs()
            self.serversocket = context.wrap_socket(socket.socket(socket.AF_INET), server_hostname=TargetPortalAddress)

            self.serversocket.connect((TargetPortalAddress, TargetPortalPortNumber))
            self.logger.info("connection succeeded %s %s" % (TargetPortalAddress, TargetPortalPortNumber))
            try:
                self.logger.info("Handshare in progress")
                self.serversocket.do_handshake()
                self.logger.info("Handshare done")
            except 	ssl.SSLError, err:
                if err.args[0] == ssl.SSL_ERROR_WANT_READ:
                    select.select([self.serversocket], [], [])
                elif err.args[0] == ssl.SSL_ERROR_WANT_WRITE:
                    select.select([], [self.serversocket], [])
                else:
                    raise
                cert = self.serversocket.getpeercert()

        self.logger.info("SSL Done %s %s" % (TargetPortalAddress, TargetPortalPortNumber))
        self.sthread=ServerThread("ServerThread",self.clientsocket,self.serversocket)
        self.sthread.start()

    def run(self):
        self.logger.info("Starting " + self.name)
        try:

            while True:
                self.logger.info("Reading from client")
                data = self.clientsocket.recv(self.chunk_size)
                if data != '':
                    self.logger.info("sending to server")
                    sent=self.serversocket.send(data[:len(data)])
                    self.logger.info("sent to server (%d)" % (sent))
                elif data == '':
                    break

        except socket.error, (value,message):
            self.logger.error('socket.error - ' + message)

        self.logger.info("Disconnected from client")

        self.clientsocket.close()
        self.serversocket.close()

def SecureTCPTunnelMain(args):
    OSVersion=sys.argv[1]
    LOG_FOLDER=sys.argv[2]
    ScriptId=sys.argv[3]
    MinPort=int(sys.argv[4])
    MaxPort=int(sys.argv[5])
    TargetPortalAddress=sys.argv[6]
    TargetPortalPortNumber=int(sys.argv[7])
    TargetNodeAddress=sys.argv[8]
    VMName=sys.argv[9]
    LOG_FILENAME = LOG_FOLDER + "/Scripts/SecureTCPTunnelLog.log"
    logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',)
    ILRTargetInfo=(TargetPortalAddress, TargetPortalPortNumber,TargetNodeAddress,ScriptId,VMName,LOG_FOLDER)
    ilr_config_file = '/etc/MicrosoftAzureBackupILR/mabilr.conf'
    port_range = (MinPort, MaxPort) # let the kernel give us a port
    server = SecureTCPTunnelServer(port_range, ILRTargetInfo, ilr_config_file)
    asyncore.loop()
if __name__ == "__main__":
    SecureTCPTunnelMain(sys.argv)
    """
    script_file_path = os.path.join(script_folder, "SecureTCPTunnel.py")
    f = open(script_file_path, "w+")
    f.write(SecureTCPTunnelCode)
    f.close()
    os.chmod(script_file_path, stat.S_IXGRP)

def get_osname_for_script(OsName):
    lowercase_osname = OsName.lower()
    if "ubuntu" in lowercase_osname:
        OsName = "Ubuntu"
    elif "debian" in lowercase_osname:
        OsName = "Debian"
    elif "centos" in lowercase_osname:
        OsName = "CentOS"
    elif "red hat" in lowercase_osname or "rhel" in lowercase_osname:
        OsName = "RHEL"
    elif "opensuse" in lowercase_osname:
        OsName = "OpenSUSE"
    elif ("suse" in lowercase_osname and "enterprise" in lowercase_osname) or "sles" in lowercase_osname:
        OsName = "SLES"
    elif "oracle" in lowercase_osname:
        OsName = "Oracle"
    return OsName

def main(argv):
    print "Microsoft Azure VM Backup - File Recovery"
    print "______________________________________________"
    try:
        (Os, Version, VersionName) = platform.linux_distribution()
        if Version.count(".") > 1:
            Versions = Version.split(".")
            Version = Versions[0] + "." + Versions[1]
        OsVersion = float(Version)
        OsName = get_osname_for_script(Os)
        SupportedOSes = ["Ubuntu", "Debian", "CentOS", "RHEL", "SLES", "OpenSUSE", "Oracle"]
        OsVersionDict = {"Ubuntu" : 12.04,
                    "Debian" : 7,
                    "CentOS" : 6.5,
                    "RHEL" : 6.7,
                    "SLES" : 12,
                    "OpenSUSE" : 42.2,
                    "Oracle" : 6.4}
        OsMajorVersionDict = {"Ubuntu" : 12,
                    "Debian" : 7,
                    "CentOS" : 6,
                    "RHEL" : 6,
                    "SLES" : 12,
                    "OpenSUSE" : 42,
                    "Oracle" : 6}
        OsMinorVersionDict = {
                    "CentOS" : 5,
                    "RHEL" : 5,
                    "OpenSUSE" : 2,
                    "Oracle" : 4}


        if OsName in SupportedOSes:
            LowerVersion = OsVersionDict[OsName]
            if not is_supported(OsVersion, LowerVersion):
                isSupportedVersion = False
                if OsName in ["CentOS", "RHEL" , "OpenSUSE", "Oracle"] and Version.count(".") == 1:
                    Versions = Version.split(".")
                    OSMajorVersion = int(Versions[0])
                    OSMinorVersion = int(Versions[1])
                    LowestMajorVersion = OsMajorVersionDict[OsName]
                    LowestMinorVersion = OsMinorVersionDict[OsName]
                    if OSMajorVersion == LowestMajorVersion and OSMinorVersion >= LowestMinorVersion:
                        isSupportedVersion = True
                if isSupportedVersion == False:
                    supported_os_msg()
                    ans=raw_input("Please press 'Y' if you still want to continue with running the script, 'N' to abort the operation. : ")
                    if ans not in ['y','Y']:
                        exit()
        else:
            print "You are running the script from an unsupported OS Version"
            supported_os_msg()
            ans=raw_input("Please press 'Y' if you still want to continue with running the script, 'N' to abort the operation. : ")
            if ans not in ['y','Y']:
                exit()
    except:
        print "OsName not recognized if your os is in the list 'Ubuntu', 'Debian', 'CentOS', 'RHEL', 'SLES', 'OpenSUSE', 'Oracle'.\n Please enter OsName as they appear in the list"
        response = raw_input("Enter you os name: ")
        OsName = get_osname_for_script(response.strip())
        if OsName not in ["Ubuntu", "Debian", "CentOS", "RHEL", "SLES", "OpenSUSE", "Oracle"]:
            print "Your Os " + OsName + " is not yet supported"
            supported_os_msg()
            exit()

    # initialize the parameters required for ILR script to run
    MinPort=5365
    MaxPort=5396
    # set up script directory and set up the logs directory
    # volume will be mounted in this directory only
    script_directory = os.getcwd()
    new_guid = str(time.strftime("%Y%m%d%H%M%S"))
    log_folder = script_directory + "/" + MachineName + "-" + new_guid
    script_folder = log_folder + "/Scripts"
    os.mkdir(log_folder)
    os.mkdir(script_folder)
    logfilename = script_folder + "/MicrosoftAzureBackupILRLogFile.log"
    global logfile
    logfile = open(logfilename,'a+')
    MABILRConfigFolder="/etc/MicrosoftAzureBackupILR"
    install_prereq_packages(OsName)
    if not os.path.exists(MABILRConfigFolder):
        os.mkdir(MABILRConfigFolder)

    log("Generating SecureTCPTunnel code")
    generate_securetcptunnel_code(script_folder)


    log("Setting ACL to Log and script Folder")
    isSetfaclInstalled = is_package_installated("setfacl")
    if isSetfaclInstalled:
        shell_cmd = 'setfacl --set="user::rwx,group::rwx,other::---" ' + log_folder
        run_shell_cmd(shell_cmd)
        shell_cmd = 'setfacl --default --set="user::rwx,group::rwx,other::---" ' + log_folder
        run_shell_cmd(shell_cmd)
        shell_cmd = 'setfacl --set="user::rwx,group::rwx,other::---" ' + script_folder
        run_shell_cmd(shell_cmd)
        shell_cmd = 'setfacl --default --set="user::rwx,group::rwx,other::---" ' + script_folder
        run_shell_cmd(shell_cmd)
    else:
        shell_cmd = 'chmod -R "ug+rwx" ' + log_folder
        run_shell_cmd(shell_cmd)
        shell_cmd = 'chmod -R "ug+rwx" ' + script_folder
        run_shell_cmd(shell_cmd)
    log("Setting ACL succeeded")
    check_for_open_SSL_TLS_v12()
    DoCleanUp = len(argv) > 0 and "clean" in argv
    ilr_params = {
        "MinPort": MinPort,
        "MaxPort": MaxPort,
        "VMName" : VMName,
        "OsNameVersion" : OsName + ";" + Version + ";",
        "MachineName" : MachineName,
        "TargetPortalAddress": TargetPortalAddress,
        "TargetPortalPortNumber": int(TargetPortalPortNumber),
        "TargetNodeAddress": TargetNodeAddress,
        "TargetUserName": TargetUserName,
        "TargetPassword": TargetPassword,
        "InitiatorChapPassword": InitiatorChapPassword,
        "ScriptId": ScriptId,
        "LogFolder": log_folder,
        "SciptFolder": script_folder,
        "DoCleanUp" : DoCleanUp,
        "IsMultiTarget" : IsMultiTarget
    }
    ILRMain(ilr_params)

if __name__ == "__main__":
    if os.getuid() != 0:
        print "Launching the ilrscript as admin"
        python_script_with_args = " ".join(sys.argv)
        os.system("sudo python " + python_script_with_args)
        exit(0)
    else:
        global VMName,MachineName,TargetPortalAddress,TargetPortalPortNumber,TargetNodeAddress,InitiatorChapPassword,ScriptId,TargetUserName,TargetPassword, IsMultiTarget
    	VMName="iaasvmcontainerv2;pt-onnx;onnx-pt-gpu"
    	MachineName="onnx-pt-gpu"
    	TargetPortalAddress="pod01-rec2.scus.backup.windowsazure.com"
    	TargetPortalPortNumber="3260"
    	TargetNodeAddress="iqn.2016-01.microsoft.azure.backup:3023779424238873068-259191-211106982664663-85620040332293.637231847969044310"
    	InitiatorChapPassword="5bcbcb546f5a62"
    	ScriptId="c8402f72-0973-4f13-a5ca-612e52fb0e8d"
    	TargetUserName="3023779424238873068-eb2b52ad-f5d8-481b-b21c-200eaf4f0830"
    	TargetPassword="UserInput012345"
    	IsMultiTarget = "0"
        main(sys.argv[1:])

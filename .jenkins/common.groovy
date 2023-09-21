// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean sameOrg=false)
{
    project.paths.construct_build_prefix()

    def sudo = platform.jenkinsLabel.contains('sles') ? '/usr/bin/sudo --preserve-env ' : ''
    String centos = platform.jenkinsLabel.contains('centos') ? 'source scl_source enable devtoolset-7' : ''
    def getDependencies = auxiliary.getLibrary('rocBLAS-internal',platform.jenkinsLabel, null, sameOrg)
    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${getDependencies}
                export PATH=/opt/rocm/bin:$PATH
                ${centos}
                ${sudo} make
            """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project)
{
    def sudo = auxiliary.sudo(platform.jenkinsLabel)

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${sudo} make run 2>&1 | tee test_log
                grep -ni error test_log
                grep -ni warning test_log
                grep -ni fail test_log
                grep -ni error test_log > test_errors
                grep -ni warning test_log >> test_errors
                grep -ni fail test_log >> test_errors
                VAR=\$(wc -l < test_errors)
                if [ \$VAR != 0 ]; then
                    exit 1
                fi
            """

    platform.runCommand(this, command)
}

return this


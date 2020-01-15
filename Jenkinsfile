@Library('rocJenkins@wgetlessverbose') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('0 7 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

import java.nio.file.Path;

rocBLAS_ExamplesCI:
{

    def rocBLAS_Examples = new rocProject('rocBLAS_Examples')
    
    def nodes = new dockerNodes(['gfx900 && ubuntu16 && internal', 'gfx906 && ubuntu16 && internal', 'gfx906 && centos7 && internal', 
    'gfx900 && centos7 && internal',  'gfx900 && sles && internal', 'gfx906 && sles && internal'], rocBLAS_Examples)

    boolean formatCheck = true

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

        def getRocBLAS = auxiliary.getLibrary('rocBLAS',platform.jenkinsLabel,'develop')
        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    ${getRocBLAS}
                    export PATH=/opt/rocm/bin:$PATH
                    make
                """

        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->
        
        def sudo = auxiliary.sudo(platform.jenkinsLabel)
        
        def getRocBLAS = auxiliary.getLibrary('rocBLAS',platform.jenkinsLabel,'develop')
        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    ${getRocBLAS}
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

    def packageCommand = null

    buildProject(rocBLAS_Examples, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}


<#
.SYNOPSIS
Deletes all vectors from a specified Pinecone index using environment variables.
.DESCRIPTION
This script reads Pinecone API key and index name from environment variables.
It retrieves the index host URL and then prompts for confirmation before deleting all vectors
using the Pinecone REST API. The PINECONE_ENVIRONMENT variable is optional.
.NOTES
Requires the following environment variables to be set:
- PINECONE_API_KEY
- PINECONE_INDEX_NAME
Optional environment variable:
- PINECONE_ENVIRONMENT (used for display only)
#>

param() # No parameters needed, uses environment variables

# --- 1. Read Environment Variables ---
$apiKey = $env:PINECONE_API_KEY
$indexName = $env:PINECONE_INDEX_NAME
$environment = $env:PINECONE_ENVIRONMENT # Read if set, but don't require it

if (-not $apiKey) {
    Write-Error "ERROR: PINECONE_API_KEY environment variable not set."
    exit 1
}
# Removed mandatory check for $environment
if (-not $indexName) {
    Write-Error "ERROR: PINECONE_INDEX_NAME environment variable not set."
    exit 1
}

Write-Host "Pinecone Configuration:"
# Display environment only if it was set
if ($environment) {
    Write-Host " - Environment: $environment (Informational)"
} else {
    Write-Host " - Environment: Not Set (Optional)"
}
Write-Host " - Index Name:  $indexName"

# --- 2. Get Index Host URL ---
# Pinecone requires requests to be sent to the specific index host.
# We can try describing the index to get its host.
$describeUri = "https://api.pinecone.io/indexes/$indexName" # Using the newer general API endpoint
$headers = @{
    "Api-Key" = $apiKey
    "Accept" = "application/json"
}

Write-Host "Fetching index details to get the host URL..."
try {
    $indexDetails = Invoke-RestMethod -Uri $describeUri -Method Get -Headers $headers -ErrorAction Stop
    $indexHost = $indexDetails.host
    if (-not $indexHost) {
         Write-Error "ERROR: Could not retrieve 'host' from index details for '$indexName'."
         exit 1
    }
     Write-Host " - Index Host:  $indexHost (This is the critical URL)"
} catch {
    Write-Error "ERROR: Failed to get index details for '$indexName'. Check API key, index name, and network connection."
    # Attempt to extract more detailed error from the response
    $errorMessage = $_.Exception.Message
    if ($_.Exception.Response) {
        try {
            $stream = $_.Exception.Response.GetResponseStream()
            $reader = New-Object System.IO.StreamReader($stream)
            $responseBody = $reader.ReadToEnd()
            $reader.Close()
            $stream.Close()
            $errorMessage += "`nAPI Response: $responseBody"
        } catch {
            $errorMessage += "`n(Could not read detailed API error response)"
        }
    }
    Write-Error $errorMessage
    exit 1
}


# --- 3. Get Vector Count (for confirmation message) ---
$statsUri = "https://$indexHost/describe_index_stats"
$vectorCount = 0
try {
    Write-Host "Fetching current vector count..."
    # Add -SkipHttpErrorCheck for newer PowerShell Core versions if needed, but try without first
    $statsResponse = Invoke-RestMethod -Uri $statsUri -Method Post -Headers $headers -Body "{}" -ContentType "application/json" -ErrorAction Stop
    # Check if totalVectorCount exists before accessing
     if ($statsResponse -and $statsResponse.PSObject.Properties['totalVectorCount']) {
        $vectorCount = $statsResponse.totalVectorCount
        Write-Host "Index '$indexName' currently contains $vectorCount vectors."

        if ($vectorCount -eq 0) {
            Write-Host "Index is already empty. No action needed."
            exit 0
        }
     } else {
         Write-Warning "Warning: Could not find 'totalVectorCount' in stats response. Proceeding without vector count confirmation."
     }
} catch {
    Write-Warning "Warning: Could not fetch index stats. Proceeding without vector count confirmation."
    # Optionally display error details for stats failure
    # $errorMessage = $_.Exception.Message
    # ... (error response reading code similar to step 2) ...
    # Write-Warning $errorMessage
}


# --- 4. CRITICAL: Confirmation Step ---
Write-Host ""
Write-Host ("=" * 40) -ForegroundColor Yellow
$countString = if ($vectorCount -gt 0) { "$vectorCount" } else { "all" }
Write-Host "WARNING: You are about to delete $countString vectors from the index '$indexName'." -ForegroundColor Yellow
Write-Host "This action is IRREVERSIBLE." -ForegroundColor Red
Write-Host ("=" * 40) -ForegroundColor Yellow
Write-Host ""

$confirmation = Read-Host -Prompt "To confirm deletion, please type the index name '$indexName' exactly"

if ($confirmation -ne $indexName) {
    Write-Host "Confirmation failed. Input '$confirmation' did not match '$indexName'. Aborting deletion." -ForegroundColor Red
    exit 1
}

# --- 5. Perform Deletion ---
$deleteUri = "https://$indexHost/vectors/delete?deleteAll=true"
Write-Host "Confirmation received. Sending delete request to $deleteUri ..."

try {
    # Use Invoke-WebRequest for DELETE as Invoke-RestMethod might have issues with empty responses sometimes
    $deleteResponse = Invoke-WebRequest -Uri $deleteUri -Method Delete -Headers $headers -ErrorAction Stop
    # Check status code for success (200 OK)
    if ($deleteResponse.StatusCode -eq 200) {
        Write-Host "Deletion command sent successfully (Status Code: $($deleteResponse.StatusCode))." -ForegroundColor Green
        Write-Host "It might take a short while for the stats to reflect the deletion."
    } else {
        Write-Error "ERROR: Delete request failed with Status Code: $($deleteResponse.StatusCode)"
        Write-Error "Response Content: $($deleteResponse.Content)"
        exit 1
    }
} catch {
    Write-Error "ERROR: Failed to send delete request to index '$indexName'."
    # Try to get more details from the error response
    $errorMessage = $_.Exception.Message
    if ($_.Exception.Response) {
        try {
            $stream = $_.Exception.Response.GetResponseStream()
            $reader = New-Object System.IO.StreamReader($stream)
            $responseBody = $reader.ReadToEnd()
            $reader.Close()
            $stream.Close()
            $errorMessage += "`nAPI Response: $responseBody"
        } catch {
             $errorMessage += "`n(Could not read detailed API error response)"
        }
    }
    Write-Error $errorMessage
    exit 1
}

Write-Host "Script finished."


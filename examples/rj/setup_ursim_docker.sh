#!/bin/bash

# URSim Docker Setup and Management Script
#
# This script automates the setup and management of a UR5e simulator (URSim)
# running in a Docker container for testing robot control software.
#
# Prerequisites:
#   - Docker installed and running
#   - User has permission to run Docker commands (or use sudo)
#
# Usage:
#   ./setup_ursim_docker.sh [command]
#
# Commands:
#   setup    - Initial setup: pull image and create network (one-time)
#   start    - Start the URSim container
#   stop     - Stop the URSim container
#   restart  - Restart the URSim container
#   status   - Check if container is running and show connection info
#   logs     - Show container logs
#   gui      - Display GUI access URLs
#   clean    - Remove container and network (WARNING: destructive)
#   help     - Show this help message

set -e  # Exit on error

# Configuration
IMAGE="universalrobots/ursim_e-series"
NETWORK_NAME="ursim_net"
SUBNET="192.168.56.0/24"
CONTAINER_NAME="ursim_ur5e"
CONTAINER_IP="192.168.56.101"

# Port mappings
PORT_VNC=5900
PORT_WEB=6080
PORT_PRIMARY=30002
PORT_SECONDARY=30003
PORT_RTDE=30004

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "$1"
}

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Create Docker network if it doesn't exist
setup_network() {
    if docker network inspect $NETWORK_NAME &> /dev/null; then
        print_warning "Network '$NETWORK_NAME' already exists"
    else
        print_info "Creating Docker network '$NETWORK_NAME'..."
        docker network create --subnet=$SUBNET $NETWORK_NAME
        print_success "Network '$NETWORK_NAME' created"
    fi
}

# Pull URSim Docker image
pull_image() {
    print_info "Pulling URSim Docker image (this may take a few minutes)..."
    docker pull $IMAGE
    print_success "Image '$IMAGE' pulled successfully"
}

# Start URSim container
start_container() {
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        # Container exists, check if it's running
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            print_warning "Container '$CONTAINER_NAME' is already running"
            show_connection_info
            return 0
        else
            # Container exists but is stopped, start it
            print_info "Starting existing container '$CONTAINER_NAME'..."
            docker start $CONTAINER_NAME
            print_success "Container '$CONTAINER_NAME' started"
            show_connection_info
            return 0
        fi
    fi

    # Create and start new container
    print_info "Creating and starting URSim container '$CONTAINER_NAME'..."
    docker run -d \
        --name $CONTAINER_NAME \
        --network $NETWORK_NAME \
        --ip $CONTAINER_IP \
        -p $PORT_VNC:$PORT_VNC \
        -p $PORT_WEB:$PORT_WEB \
        -p $PORT_PRIMARY:$PORT_PRIMARY \
        -p $PORT_SECONDARY:$PORT_SECONDARY \
        -p $PORT_RTDE:$PORT_RTDE \
        $IMAGE

    print_success "Container '$CONTAINER_NAME' created and started"
    print_info ""
    print_info "Waiting for URSim to initialize (this may take 10-15 seconds)..."
    sleep 5
    show_connection_info
}

# Stop URSim container
stop_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "Container '$CONTAINER_NAME' is not running"
        return 0
    fi

    print_info "Stopping container '$CONTAINER_NAME'..."
    docker stop $CONTAINER_NAME
    print_success "Container '$CONTAINER_NAME' stopped"
}

# Restart URSim container
restart_container() {
    print_info "Restarting container '$CONTAINER_NAME'..."
    docker restart $CONTAINER_NAME
    print_success "Container '$CONTAINER_NAME' restarted"
    sleep 5
    show_connection_info
}

# Check container status
check_status() {
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_success "Container '$CONTAINER_NAME' is running"
        show_connection_info
        return 0
    elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "Container '$CONTAINER_NAME' exists but is stopped"
        print_info "Run './setup_ursim_docker.sh start' to start it"
        return 1
    else
        print_warning "Container '$CONTAINER_NAME' does not exist"
        print_info "Run './setup_ursim_docker.sh setup' first, then './setup_ursim_docker.sh start'"
        return 1
    fi
}

# Show container logs
show_logs() {
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_error "Container '$CONTAINER_NAME' does not exist"
        return 1
    fi

    print_info "Showing logs for container '$CONTAINER_NAME' (press Ctrl+C to exit)..."
    docker logs -f $CONTAINER_NAME
}

# Display connection information
show_connection_info() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  URSim Connection Information"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  Web GUI (easiest):"
    echo "    http://localhost:$PORT_WEB/vnc.html"
    echo ""
    echo "  VNC Client:"
    echo "    vnc://localhost:$PORT_VNC"
    echo ""
    echo "  RTDE Interface (for robot control):"
    echo "    IP: $CONTAINER_IP"
    echo "    Port: $PORT_RTDE"
    echo ""
    echo "  Additional Interfaces:"
    echo "    Primary:   $CONTAINER_IP:$PORT_PRIMARY"
    echo "    Secondary: $CONTAINER_IP:$PORT_SECONDARY"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "  1. Open web GUI: http://localhost:$PORT_WEB/vnc.html"
    echo "  2. Click 'ON' button to power on the robot"
    echo "  3. Wait ~30 seconds for initialization"
    echo "  4. Click 'START' to start the robot program"
    echo "  5. Run: python examples/rj/spacemouse_teleop_ur5e.py"
    echo ""
}

# Clean up container and network
clean_all() {
    print_warning "This will remove the container and network. Are you sure? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_info "Cancelled"
        return 0
    fi

    # Stop and remove container
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_info "Removing container '$CONTAINER_NAME'..."
        docker rm -f $CONTAINER_NAME
        print_success "Container removed"
    fi

    # Remove network
    if docker network inspect $NETWORK_NAME &> /dev/null; then
        print_info "Removing network '$NETWORK_NAME'..."
        docker network rm $NETWORK_NAME
        print_success "Network removed"
    fi

    print_success "Cleanup complete"
}

# Show help message
show_help() {
    echo "URSim Docker Setup and Management Script"
    echo ""
    echo "Usage: ./setup_ursim_docker.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup    - Initial setup: pull image and create network (one-time)"
    echo "  start    - Start the URSim container"
    echo "  stop     - Stop the URSim container"
    echo "  restart  - Restart the URSim container"
    echo "  status   - Check if container is running and show connection info"
    echo "  logs     - Show container logs (follow mode)"
    echo "  gui      - Display GUI access URLs"
    echo "  clean    - Remove container and network (WARNING: destructive)"
    echo "  help     - Show this help message"
    echo ""
    echo "Example workflow:"
    echo "  ./setup_ursim_docker.sh setup    # One-time setup"
    echo "  ./setup_ursim_docker.sh start    # Start URSim"
    echo "  ./setup_ursim_docker.sh status   # Check status"
    echo "  ./setup_ursim_docker.sh stop     # Stop when done"
    echo ""
}

# Main script logic
main() {
    check_docker

    case "${1:-help}" in
        setup)
            print_info "Setting up URSim Docker environment..."
            setup_network
            pull_image
            print_success "Setup complete!"
            print_info ""
            print_info "Next step: Run './setup_ursim_docker.sh start' to start URSim"
            ;;
        start)
            start_container
            ;;
        stop)
            stop_container
            ;;
        restart)
            restart_container
            ;;
        status)
            check_status
            ;;
        logs)
            show_logs
            ;;
        gui)
            show_connection_info
            ;;
        clean)
            clean_all
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

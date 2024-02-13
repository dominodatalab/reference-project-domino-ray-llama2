import logging
import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from google.protobuf.json_format import MessageToDict

from ray.autoscaler.v2.instance_manager.common import InstanceUtil
from ray.autoscaler.v2.instance_manager.config import (
    AutoscalingConfig,
    InstanceReconcileConfig,
)
from ray.autoscaler.v2.instance_manager.instance_manager import InstanceManager
from ray.autoscaler.v2.instance_manager.node_provider import (
    CloudInstance,
    CloudInstanceId,
    CloudInstanceProviderError,
    ICloudInstanceProvider,
    LaunchNodeError,
    TerminateNodeError,
)
from ray.autoscaler.v2.instance_manager.ray_installer import RayInstallError
from ray.autoscaler.v2.scheduler import IResourceScheduler
from ray.core.generated.autoscaler_pb2 import (
    AutoscalingState,
    ClusterResourceState,
    NodeState,
    NodeStatus,
)
from ray.core.generated.instance_manager_pb2 import GetInstanceManagerStateRequest
from ray.core.generated.instance_manager_pb2 import Instance as IMInstance
from ray.core.generated.instance_manager_pb2 import (
    InstanceUpdateEvent as IMInstanceUpdateEvent,
)
from ray.core.generated.instance_manager_pb2 import (
    StatusCode,
    UpdateInstanceManagerStateRequest,
)

logger = logging.getLogger(__name__)


class Reconciler:
    """
    A singleton class that reconciles the instance states of the instance manager
    for autoscaler.

    """

    @staticmethod
    def reconcile(
        instance_manager: InstanceManager,
        scheduler: IResourceScheduler,
        cloud_provider: ICloudInstanceProvider,
        ray_cluster_resource_state: ClusterResourceState,
        non_terminated_cloud_instances: Dict[CloudInstanceId, CloudInstance],
        cloud_provider_errors: List[CloudInstanceProviderError],
        ray_install_errors: List[RayInstallError],
        autoscaling_config: AutoscalingConfig,
        _logger: Optional[logging.Logger] = None,
    ) -> AutoscalingState:
        """
        The reconcile method computes InstanceUpdateEvents for the instance manager
        by:

        1. Reconciling the instance manager's instances with external states like
        the cloud provider's, the ray cluster's states, the ray installer's results.
        It performs "passive" status transitions for the instances (where the status
        transition should only be reflecting the external states of the cloud provider
        and the ray cluster, and should not be actively changing them)

        2. Stepping the instances to the active states by computing instance status
        transitions that are needed and updating the instance manager's state.
        These transitions should be "active" where the transitions have side effects
        (through InstanceStatusSubscriber) to the cloud provider and the ray cluster.

        Args:
            instance_manager: The instance manager to reconcile.
            ray_cluster_resource_state: The ray cluster's resource state.
            non_terminated_cloud_instances: The non-terminated cloud instances from
                the cloud provider.
            cloud_provider_errors: The errors from the cloud provider.
            ray_install_errors: The errors from RayInstaller.

        """
        Reconciler._sync_from(
            instance_manager=instance_manager,
            ray_nodes=ray_cluster_resource_state.node_states,
            non_terminated_cloud_instances=non_terminated_cloud_instances,
            cloud_provider_errors=cloud_provider_errors,
            ray_install_errors=ray_install_errors,
        )

        autoscaling_state = AutoscalingState()
        Reconciler._step_next(
            autoscaling_state=autoscaling_state,
            instance_manager=instance_manager,
            scheduler=scheduler,
            cloud_provider=cloud_provider,
            ray_cluster_resource_state=ray_cluster_resource_state,
            non_terminated_cloud_instances=non_terminated_cloud_instances,
            autoscaling_config=autoscaling_config,
            _logger=_logger,
        )
        return autoscaling_state

    @staticmethod
    def _sync_from(
        instance_manager: InstanceManager,
        ray_nodes: List[NodeState],
        non_terminated_cloud_instances: Dict[CloudInstanceId, CloudInstance],
        cloud_provider_errors: List[CloudInstanceProviderError],
        ray_install_errors: List[RayInstallError],
    ):
        """
        Reconcile the instance states of the instance manager from external states like
        the cloud provider's, the ray cluster's states, the ray installer's results,
        etc.

        For each instance, we try to figure out if we need to transition the instance
        status to a new status, and if so, what the new status should be.

        These transitions should be purely "passive", meaning they should only be
        reflecting the external states of the cloud provider and the ray cluster,
        and should not be actively changing the states of the cloud provider or the ray
        cluster.

        More specifically, we will reconcile status transitions for:
            1.  QUEUED/REQUESTED -> ALLOCATED:
                When a instance with launch request id (indicating a previous launch
                request was made) could be assigned to an unassigned cloud instance
                of the same instance type.
            2.  REQUESTED -> ALLOCATION_FAILED:
                When there's an error from the cloud provider for launch failure so
                that the instance becomes ALLOCATION_FAILED.
            3.  * -> RAY_RUNNING:
                When a ray node on a cloud instance joins the ray cluster, we will
                transition the instance to RAY_RUNNING.
            4.  * -> TERMINATED:
                When the cloud instance is already terminated, we will transition the
                instance to TERMINATED.
            5.  TERMINATING -> TERMINATION_FAILED:
                When there's an error from the cloud provider for termination failure.
            6.  * -> RAY_STOPPED:
                When ray was stopped on the cloud instance, we will transition the
                instance to RAY_STOPPED.
            7.  * -> RAY_INSTALL_FAILED:
                When there's an error from RayInstaller.

        Args:
            instance_manager: The instance manager to reconcile.
            ray_nodes: The ray cluster's states of ray nodes.
            non_terminated_cloud_instances: The non-terminated cloud instances from
                the cloud provider.
            cloud_provider_errors: The errors from the cloud provider.
            ray_install_errors: The errors from RayInstaller.

        """

        # Handle 1 & 2 for cloud instance allocation.
        Reconciler._handle_cloud_instance_allocation(
            instance_manager,
            non_terminated_cloud_instances,
            cloud_provider_errors,
        )
        Reconciler._handle_cloud_instance_terminated(
            instance_manager, non_terminated_cloud_instances
        )
        Reconciler._handle_ray_status_transition(instance_manager, ray_nodes)

        Reconciler._handle_cloud_instance_termination_errors(
            instance_manager, cloud_provider_errors
        )

        Reconciler._handle_ray_install_failed(instance_manager, ray_install_errors)

    @staticmethod
    def _step_next(
        autoscaling_state: AutoscalingState,
        instance_manager: InstanceManager,
        scheduler: IResourceScheduler,
        cloud_provider: ICloudInstanceProvider,
        ray_cluster_resource_state: ClusterResourceState,
        non_terminated_cloud_instances: Dict[CloudInstanceId, CloudInstance],
        autoscaling_config: AutoscalingConfig,
        _logger: Optional[logging.Logger] = None,
    ):
        """
        Step the reconciler to the next state by computing instance status transitions
        that are needed and updating the instance manager's state.

        Specifically, we will:
            1. Shut down leak cloud instances
                Leaked cloud instances that are not managed by the instance manager.
            2. Terminating instances with ray stopped or ray install failure.
            3. Scale up/down the cluster:
              (* -> RAY_STOPPING)
                b. Extra cloud due to max nodes config.
                c. Cloud instances with outdated configs.
            4. Create new instances
              (new QUEUED)
                Create new instances based on the IResourceScheduler's decision for
                scaling up.
            5. Request cloud provider to launch new instances.
              (QUEUED -> REQUESTED)
            6. Install ray
              (ALLOCATED -> RAY_INSTALLING)
                When ray could be installed and launched.
            7. Handle any stuck instances with timeouts.

        Args:
            instance_manager: The instance manager to reconcile.
            scheduler: The resource scheduler to make scaling decisions.
            ray_cluster_resource_state: The ray cluster's resource state.
            non_terminated_cloud_instances: The non-terminated cloud instances from
                the cloud provider.

        """

        Reconciler._handle_extra_cloud_instances(
            instance_manager, non_terminated_cloud_instances
        )

        Reconciler._handle_stuck_instances(
            instance_manager=instance_manager,
            reconcile_config=autoscaling_config.get_instance_reconcile_config(),
            _logger=_logger or logger,
        )

        Reconciler._handle_instances_launch(
            instance_manager=instance_manager, autoscaling_config=autoscaling_config
        )

        Reconciler._terminate_instances(instance_manager=instance_manager)
        if not autoscaling_config.skip_ray_install():
            Reconciler._install_ray(
                instance_manager=instance_manager,
                non_terminated_cloud_instances=non_terminated_cloud_instances,
            )

    #######################################################
    # Utility methods for reconciling instance states.
    #######################################################

    @staticmethod
    def _handle_cloud_instance_allocation(
        instance_manager: InstanceManager,
        non_terminated_cloud_instances: Dict[CloudInstanceId, CloudInstance],
        cloud_provider_errors: List[CloudInstanceProviderError],
    ):
        im_instances, version = Reconciler._get_im_instances(instance_manager)
        updates = {}

        # Compute intermediate states.
        instances_with_launch_requests: List[IMInstance] = [
            instance for instance in im_instances if instance.launch_request_id
        ]
        assigned_cloud_instance_ids: Set[CloudInstanceId] = {
            instance.cloud_instance_id for instance in im_instances
        }
        launch_errors: Dict[str, LaunchNodeError] = {
            error.request_id: error
            for error in cloud_provider_errors
            if isinstance(error, LaunchNodeError)
        }
        unassigned_cloud_instances_by_type: Dict[
            str, List[CloudInstance]
        ] = defaultdict(list)

        for cloud_instance_id, cloud_instance in non_terminated_cloud_instances.items():
            if cloud_instance_id not in assigned_cloud_instance_ids:
                unassigned_cloud_instances_by_type[cloud_instance.node_type].append(
                    cloud_instance
                )

        # Sort the request instance by the increasing request time.
        instances_with_launch_requests.sort(
            key=lambda instance: InstanceUtil.get_status_transition_times_ns(
                instance, IMInstance.REQUESTED
            )
        )

        # For each instance, try to allocate or fail the allocation.
        for instance in instances_with_launch_requests:
            # Try allocate or fail with errors.
            update_event = Reconciler._try_resolve_pending_allocation(
                instance, unassigned_cloud_instances_by_type, launch_errors
            )
            if not update_event:
                continue

            logger.debug(
                "Updating {}({}) with {}".format(
                    instance.instance_id,
                    IMInstance.InstanceStatus.Name(instance.status),
                    MessageToDict(update_event),
                )
            )
            updates[instance.instance_id] = update_event

        # Update the instance manager for the events.
        Reconciler._update_instance_manager(instance_manager, version, updates)

    @staticmethod
    def _try_resolve_pending_allocation(
        im_instance: IMInstance,
        unassigned_cloud_instances_by_type: Dict[str, List[CloudInstance]],
        launch_errors: Dict[str, LaunchNodeError],
    ) -> Optional[IMInstanceUpdateEvent]:
        """
        Allocate, or fail the cloud instance allocation for the instance.

        Args:
            im_instance: The instance to allocate or fail.
            unassigned_cloud_instances_by_type: The unassigned cloud instances by type.
            launch_errors: The launch errors from the cloud provider.

        Returns:
            Instance update to ALLOCATED: if there's a matching unassigned cloud
                instance with the same type.
            Instance update to ALLOCATION_FAILED: if the instance allocation failed
                with errors.
            None: if there's no update.

        """
        unassigned_cloud_instance = None

        # Try to allocate an unassigned cloud instance.
        # TODO(rickyx): We could also look at the launch request id
        # on the cloud node and the im instance later once all node providers
        # support request id. For now, we only look at the instance type.
        if len(unassigned_cloud_instances_by_type.get(im_instance.instance_type, [])):
            unassigned_cloud_instance = unassigned_cloud_instances_by_type[
                im_instance.instance_type
            ].pop()

        if unassigned_cloud_instance:
            return IMInstanceUpdateEvent(
                instance_id=im_instance.instance_id,
                new_instance_status=IMInstance.ALLOCATED,
                cloud_instance_id=unassigned_cloud_instance.cloud_instance_id,
            )

        # If there's a launch error, transition to ALLOCATION_FAILED.
        launch_error = launch_errors.get(im_instance.launch_request_id)
        if launch_error and launch_error.node_type == im_instance.instance_type:
            return IMInstanceUpdateEvent(
                instance_id=im_instance.instance_id,
                new_instance_status=IMInstance.ALLOCATION_FAILED,
                details=launch_error.details,
            )
        # No update.
        return None

    @staticmethod
    def _handle_ray_install_failed(
        instance_manager: InstanceManager, ray_install_errors: List[RayInstallError]
    ):

        instances, version = Reconciler._get_im_instances(instance_manager)
        updates = {}

        # Get all instances with RAY_INSTALLING status.
        instances_with_ray_installing = {
            instance.instance_id: instance
            for instance in instances
            if instance.status == IMInstance.RAY_INSTALLING
        }

        install_errors = {error.im_instance_id: error for error in ray_install_errors}

        # For each instance with RAY_INSTALLING status, check if there's any
        # install error.
        for instance_id, instance in instances_with_ray_installing.items():
            install_error = install_errors.get(instance_id)
            if install_error:
                updates[instance_id] = IMInstanceUpdateEvent(
                    instance_id=instance_id,
                    new_instance_status=IMInstance.RAY_INSTALL_FAILED,
                    details=install_error.details,
                )
                logger.debug(
                    "Updating {}({}) with {}".format(
                        instance_id,
                        IMInstance.InstanceStatus.Name(instance.status),
                        MessageToDict(updates[instance_id]),
                    )
                )

        # Update the instance manager for the events.
        Reconciler._update_instance_manager(instance_manager, version, updates)

    @staticmethod
    def _handle_cloud_instance_terminated(
        instance_manager: InstanceManager,
        non_terminated_cloud_instances: Dict[CloudInstanceId, CloudInstance],
    ):
        """
        For any IM (instance manager) instance with a cloud node id, if the mapped
        cloud instance is no longer running, transition the instance to TERMINATED.

        Args:
            instance_manager: The instance manager to reconcile.
            non_terminated_cloud_instances: The non-terminated cloud instances from
                the cloud provider.
        """
        updates = {}
        instances, version = Reconciler._get_im_instances(instance_manager)

        instances_with_cloud_instance_assigned = {
            instance.cloud_instance_id: instance
            for instance in instances
            if instance.cloud_instance_id
        }

        for (
            cloud_instance_id,
            instance,
        ) in instances_with_cloud_instance_assigned.items():
            if cloud_instance_id in non_terminated_cloud_instances.keys():
                # The cloud instance is still running.
                continue

            # The cloud instance is terminated.
            updates[instance.instance_id] = IMInstanceUpdateEvent(
                instance_id=instance.instance_id,
                new_instance_status=IMInstance.TERMINATED,
                details=f"Cloud instance {cloud_instance_id} is terminated.",
            )

            logger.debug(
                "Updating {}({}) with {}".format(
                    instance.instance_id,
                    IMInstance.InstanceStatus.Name(instance.status),
                    MessageToDict(updates[instance.instance_id]),
                )
            )

        Reconciler._update_instance_manager(instance_manager, version, updates)

    @staticmethod
    def _handle_cloud_instance_termination_errors(
        instance_manager: InstanceManager,
        cloud_provider_errors: List[CloudInstanceProviderError],
    ):
        """
        If any TERMINATING instances have termination errors, transition the instance to
        TERMINATION_FAILED.

        We will retry the termination for the TERMINATION_FAILED instances in the next
        reconciler step.

        Args:
            instance_manager: The instance manager to reconcile.
            cloud_provider_errors: The errors from the cloud provider.

        """
        instances, version = Reconciler._get_im_instances(instance_manager)
        updates = {}

        termination_errors = {
            error.cloud_instance_id: error
            for error in cloud_provider_errors
            if isinstance(error, TerminateNodeError)
        }

        terminating_instances_by_cloud_instance_id = {
            instance.cloud_instance_id: instance
            for instance in instances
            if instance.status == IMInstance.TERMINATING
        }

        for cloud_instance_id, failure in termination_errors.items():
            instance = terminating_instances_by_cloud_instance_id.get(cloud_instance_id)
            if not instance:
                # The instance is no longer in TERMINATING status.
                continue

            updates[instance.instance_id] = IMInstanceUpdateEvent(
                instance_id=instance.instance_id,
                new_instance_status=IMInstance.TERMINATION_FAILED,
                details=failure.details,
            )
            logger.debug(
                "Updating {}({}) with {}".format(
                    instance.instance_id,
                    IMInstance.InstanceStatus.Name(instance.status),
                    MessageToDict(updates[instance.instance_id]),
                )
            )

        Reconciler._update_instance_manager(instance_manager, version, updates)

    @staticmethod
    def _get_im_instances(
        instance_manager: InstanceManager,
    ) -> Tuple[List[IMInstance], int]:
        reply = instance_manager.get_instance_manager_state(
            request=GetInstanceManagerStateRequest()
        )
        assert reply.status.code == StatusCode.OK
        im_state = reply.state
        return im_state.instances, im_state.version

    @staticmethod
    def _update_instance_manager(
        instance_manager: InstanceManager,
        version: int,
        updates: Dict[str, IMInstanceUpdateEvent],
    ) -> None:
        if not updates:
            return

        updates = list(updates.values()) or []

        reply = instance_manager.update_instance_manager_state(
            request=UpdateInstanceManagerStateRequest(
                expected_version=version,
                updates=updates,
            )
        )
        # TODO: While it's possible that a version mismatch
        # happens, or some other failures could happen. But given
        # the current implementation:
        #   1. There's only 1 writer (the reconciler) for updating the instance
        #       manager states, so there shouldn't be version mismatch.
        #   2. Any failures in one reconciler step should be caught at a higher
        #       level and be retried in the next reconciler step. If the IM
        #       fails to be updated, we don't have sufficient info to handle it
        #       here.
        assert (
            reply.status.code == StatusCode.OK
        ), f"Failed to update instance manager: {reply}"

    @staticmethod
    def _handle_ray_status_transition(
        instance_manager: InstanceManager, ray_nodes: List[NodeState]
    ):
        """
        Handle the ray status transition for the instance manager.

        If a new ray node running on the instance, transition it to RAY_RUNNING.
        If a ray node stopped, transition it to RAY_STOPPED.
        If a ray node is draining, transition it to RAY_STOPPING.

        Args:
            instance_manager: The instance manager to reconcile.
            ray_nodes: The ray cluster's states of ray nodes.
        """
        instances, version = Reconciler._get_im_instances(instance_manager)
        updates = {}

        im_instances_by_cloud_instance_id = {
            i.cloud_instance_id: i for i in instances if i.cloud_instance_id
        }
        ray_nodes_by_cloud_instance_id = {}
        for n in ray_nodes:
            if n.instance_id:
                ray_nodes_by_cloud_instance_id[n.instance_id] = n
            else:
                # This should only happen to a ray node that's not managed by us.
                logger.warning(
                    f"Ray node {n.node_id.decode()} has no instance id. "
                    "This only happens to a ray node that's not managed by autoscaler. "
                    "If not, please file a bug at https://github.com/ray-project/ray"
                )

        for cloud_instance_id, ray_node in ray_nodes_by_cloud_instance_id.items():
            if cloud_instance_id not in im_instances_by_cloud_instance_id:
                # This is a ray node that's not managed by the instance manager.
                # or we haven't discovered the instance yet. There's nothing
                # much we could do here.
                logger.info(
                    f"Ray node {ray_node.node_id.decode()} has no matching instance in "
                    f"instance manager with cloud instance id={cloud_instance_id}."
                )
                continue

            im_instance = im_instances_by_cloud_instance_id[cloud_instance_id]
            reconciled_im_status = Reconciler._reconciled_im_status_from_ray_status(
                ray_node.status, im_instance.status
            )
            if not reconciled_im_status:
                logger.error(
                    "Failed to reconcile from ray status: "
                    f"im_instance={im_instance.instance_id} "
                    f"with cloud instance id={cloud_instance_id}, "
                    f"cur_status={IMInstance.InstanceStatus.Name(im_instance.status)}, "
                    f"ray status={NodeStatus.Name(ray_node.status)}"
                )
                continue

            if reconciled_im_status != im_instance.status:
                updates[im_instance.instance_id] = IMInstanceUpdateEvent(
                    instance_id=im_instance.instance_id,
                    new_instance_status=reconciled_im_status,
                    details="Reconciled from ray node status "
                    f"{NodeStatus.Name(ray_node.status)} "
                    f"for ray node {ray_node.node_id.decode()}",
                    ray_node_id=ray_node.node_id.decode(),
                )
                logger.debug(
                    "Updating {}({}) with {}.".format(
                        im_instance.instance_id,
                        IMInstance.InstanceStatus.Name(im_instance.status),
                        MessageToDict(updates[im_instance.instance_id]),
                    )
                )

        Reconciler._update_instance_manager(instance_manager, version, updates)

    @staticmethod
    def _reconciled_im_status_from_ray_status(
        ray_status: NodeStatus, cur_im_status: IMInstance.InstanceStatus
    ) -> Optional["IMInstance.InstanceStatus"]:
        """
        Reconcile the instance status from the ray node status.
        Args:
            ray_status: the current ray node status.
            cur_im_status: the current IM instance status.
        Returns:
            The reconciled IM instance status, or None if no reconciliation
            could be done,  e.g. the ray node has an undefined status.
        """
        reconciled_im_status = None
        if ray_status in [NodeStatus.RUNNING, NodeStatus.IDLE]:
            reconciled_im_status = IMInstance.RAY_RUNNING
        elif ray_status == NodeStatus.DEAD:
            reconciled_im_status = IMInstance.RAY_STOPPED
        elif ray_status == NodeStatus.DRAINING:
            reconciled_im_status = IMInstance.RAY_STOPPING
        else:
            return None

        if (
            cur_im_status == reconciled_im_status
            or cur_im_status
            in InstanceUtil.get_reachable_statuses(reconciled_im_status)
        ):
            # No need to reconcile if the instance is already in the reconciled status
            # or has already transitioned beyond it.
            return cur_im_status

        return reconciled_im_status

    @staticmethod
    def _handle_instances_launch(
        instance_manager: InstanceManager, autoscaling_config: AutoscalingConfig
    ):

        instances, version = Reconciler._get_im_instances(instance_manager)

        queued_instances = []
        requested_instances = []
        allocated_instances = []

        for instance in instances:
            if instance.status == IMInstance.QUEUED:
                queued_instances.append(instance)
            elif instance.status == IMInstance.REQUESTED:
                requested_instances.append(instance)
            elif instance.cloud_instance_id:
                allocated_instances.append(instance)

        if not queued_instances:
            # No QUEUED instances
            return

        to_launch = Reconciler._compute_to_launch(
            queued_instances,
            requested_instances,
            allocated_instances,
            autoscaling_config.get_upscaling_speed(),
            autoscaling_config.get_max_concurrent_launches(),
        )

        # Transition the instances to REQUESTED for instance launcher to
        # launch them.
        updates = {}
        for instance in to_launch:
            # Reuse launch request id for any QUEUED instances that have been
            # requested before due to retry.
            launch_request_id = (
                str(time.time_ns())
                if len(instance.launch_request_id) == 0
                else instance.launch_request_id
            )
            updates[instance.instance_id] = IMInstanceUpdateEvent(
                instance_id=instance.instance_id,
                new_instance_status=IMInstance.REQUESTED,
                launch_request_id=launch_request_id,
            )
            logger.debug(
                "Updating {}({}) with {}".format(
                    instance.instance_id,
                    IMInstance.InstanceStatus.Name(instance.status),
                    MessageToDict(updates[instance.instance_id]),
                )
            )

        Reconciler._update_instance_manager(instance_manager, version, updates)

    @staticmethod
    def _compute_to_launch(
        queued_instances: List[IMInstance],
        requested_instances: List[IMInstance],
        allocated_instances: List[IMInstance],
        upscaling_speed: float,
        max_concurrent_launches: int,
    ) -> List[IMInstance]:
        def _group_by_type(instances):
            instances_by_type = defaultdict(list)
            for instance in instances:
                instances_by_type[instance.instance_type].append(instance)
            return instances_by_type

        # Sort the instances by the time they were queued.
        def _sort_by_earliest_queued(instance: IMInstance) -> List[int]:
            queue_times = InstanceUtil.get_status_transition_times_ns(
                instance, IMInstance.QUEUED
            )
            return sorted(queue_times)

        queued_instances_by_type = _group_by_type(queued_instances)
        requested_instances_by_type = _group_by_type(requested_instances)
        allocated_instances_by_type = _group_by_type(allocated_instances)

        total_num_requested_to_launch = len(requested_instances)
        all_to_launch = []

        for (
            instance_type,
            queued_instances_for_type,
        ) in queued_instances_by_type.items():
            requested_instances_for_type = requested_instances_by_type.get(
                instance_type, []
            )
            allocated_instances_for_type = allocated_instances_by_type.get(
                instance_type, []
            )

            num_desired_to_upscale = max(
                1,
                math.ceil(
                    upscaling_speed
                    * (
                        len(requested_instances_for_type)
                        + len(allocated_instances_for_type)
                    )
                ),
            )

            # Enforce global limit, at most we can launch `max_concurrent_launches`
            num_to_launch = min(
                max_concurrent_launches - total_num_requested_to_launch,
                num_desired_to_upscale,
            )

            # Cap both ends 0 <= num_to_launch <= num_queued
            num_to_launch = max(0, num_to_launch)
            num_to_launch = min(len(queued_instances_for_type), num_to_launch)

            to_launch = sorted(queued_instances_for_type, key=_sort_by_earliest_queued)[
                :num_to_launch
            ]

            all_to_launch.extend(to_launch)
            total_num_requested_to_launch += num_to_launch

        return all_to_launch

    @staticmethod
    def _handle_stuck_instances(
        instance_manager: InstanceManager,
        reconcile_config: InstanceReconcileConfig,
        _logger: logging.Logger,
    ):
        """
        Handle stuck instances with timeouts.

        Instances could be stuck in the following status and needs to be updated:
            - REQUESTED: cloud provider is slow/fails to launch instances.
            - ALLOCATED: ray fails to be started on the instance.
            - RAY_INSTALLING: ray fails to be installed on the instance.
            - TERMINATING: cloud provider is slow/fails to terminate instances.

        Instances could be in the following status which could be unbounded or
        transient, and we don't have a timeout mechanism to handle them. We would
        warn if they are stuck for too long:
            - RAY_STOPPING: ray taking time to drain.
            - QUEUED: cloud provider is slow to launch instances, resulting in long
                queue.

            Reconciler should handle below statuses, if not, could be slow
                reconcilation loop or a bug:
            - RAY_INSTALL_FAILED
            - RAY_STOPPED
            - TERMINATION_FAILED


        Args:
            instance_manager: The instance manager to reconcile.
            reconcile_config: The instance reconcile config.
            _logger: The logger to log the warning messages. It's used for testing.

        """
        instances, version = Reconciler._get_im_instances(instance_manager)

        instances_by_status = defaultdict(list)
        for instance in instances:
            instances_by_status[instance.status].append(instance)

        # Fail or retry the cloud instance allocation if it's stuck
        # in the REQUESTED state.
        im_updates = {}
        im_updates.update(
            Reconciler._handle_requested_timeout(
                instances_by_status[IMInstance.REQUESTED],
                request_status_timeout_s=reconcile_config.request_status_timeout_s,
                max_num_request_to_allocate=reconcile_config.max_num_retry_request_to_allocate,  # noqa
            )
        )

        # Handle the timeout for the following statuses.
        for cur_status, next_status, timeout in [
            # Leaked ALLOCATED instances should be terminated.
            # This usually happens when ray fails to be started on the instance, so
            # it's unable to be RAY_RUNNING after a long time.
            (
                IMInstance.ALLOCATED,
                IMInstance.TERMINATING,
                reconcile_config.allocate_status_timeout_s,
            ),
            # Fail the installation if it's stuck in RAY_INSTALLING for too long.
            # If RAY_INSTALLING is stuck for too long, it's likely that the instance
            # is not able to install ray, so we should also fail the installation.
            (
                IMInstance.RAY_INSTALLING,
                IMInstance.RAY_INSTALL_FAILED,
                reconcile_config.ray_install_status_timeout_s,
            ),
            # If we tried to terminate the instance, but it doesn't terminate (disappear
            # from the cloud provider) after a long time, we fail the termination.
            # This will trigger another attempt to terminate the instance.
            (
                IMInstance.TERMINATING,
                IMInstance.TERMINATION_FAILED,
                reconcile_config.terminating_status_timeout_s,
            ),
        ]:
            im_updates.update(
                Reconciler._handle_timeout(
                    instances_by_status[cur_status],
                    status_timeout_s=timeout,
                    cur_status=cur_status,
                    new_status=next_status,
                )
            )

        # These statues could be unbounded or transient, and we don't have a timeout
        # mechanism to handle them. We only warn if they are stuck for too long.
        for status in [
            # Ray taking time to drain. We could also have a timeout when Drain protocol
            # supports timeout.
            IMInstance.RAY_STOPPING,
            # These should just be transient, we will terminate instances with this
            # status in the next reconciler step.
            IMInstance.RAY_INSTALL_FAILED,
            IMInstance.RAY_STOPPED,
            IMInstance.TERMINATION_FAILED,
            # Instances could be in the QUEUED status for a long time if the cloud
            # provider is slow to launch instances.
            IMInstance.QUEUED,
        ]:
            Reconciler._warn_stuck_instances(
                instances_by_status[status],
                status=status,
                warn_interval_s=reconcile_config.transient_status_warn_interval_s,
                logger=_logger,
            )

        Reconciler._update_instance_manager(instance_manager, version, im_updates)

    @staticmethod
    def _handle_requested_timeout(
        instances: List[IMInstance],
        request_status_timeout_s: int,
        max_num_request_to_allocate: int,
    ) -> Dict[str, IMInstanceUpdateEvent]:
        """Change REQUESTED instances to QUEUED if they are stuck in REQUESTED state,
        or fail the allocation (ALLOCATION_FAILED) if retry too many times.
        """

        def _retry_or_fail_allocation(
            instance: IMInstance,
        ) -> Optional[IMInstanceUpdateEvent]:
            all_request_times_ns = sorted(
                InstanceUtil.get_status_transition_times_ns(
                    instance, select_instance_status=IMInstance.REQUESTED
                )
            )
            assert len(all_request_times_ns) >= 1, (
                f"instance {instance.instance_id} has {len(all_request_times_ns)} "
                f"{IMInstance.InstanceStatus.Name(IMInstance.REQUESTED)} status"
            )
            # Retry the allocation if we have waited for too long.
            last_request_time_ns = all_request_times_ns[-1]
            if time.time_ns() - last_request_time_ns <= request_status_timeout_s * 1e9:
                # We have not waited for too long. Be patient.
                return None

            # Fail the allocation if we have tried too many times.
            if len(all_request_times_ns) >= max_num_request_to_allocate:
                return IMInstanceUpdateEvent(
                    instance_id=instance.instance_id,
                    new_instance_status=IMInstance.ALLOCATION_FAILED,
                    details=(
                        "Failed to allocate cloud instance after "
                        f"{len(all_request_times_ns)} attempts"
                    ),
                )

            # Retry the allocation if we could by transitioning to QUEUED again.
            return IMInstanceUpdateEvent(
                instance_id=instance.instance_id,
                new_instance_status=IMInstance.QUEUED,
                details=f"QUEUED again after timeout={request_status_timeout_s}s",
            )

        updates = {}
        for ins in instances:
            update = _retry_or_fail_allocation(ins)
            if update:
                logger.info(
                    "Updating {}({}) with {}".format(
                        ins.instance_id,
                        IMInstance.InstanceStatus.Name(ins.status),
                        MessageToDict(update),
                    )
                )
                updates[ins.instance_id] = update

        return updates

    @staticmethod
    def _handle_timeout(
        instances: List[IMInstance],
        status_timeout_s: int,
        cur_status: IMInstance.InstanceStatus,
        new_status: IMInstance.InstanceStatus,
    ) -> Dict[str, IMInstanceUpdateEvent]:
        """Change any instances that have not transitioned to the new status
        to the new status."""
        updates = {}
        for instance in instances:
            status_times_ns = InstanceUtil.get_status_transition_times_ns(
                instance, select_instance_status=cur_status
            )
            assert len(status_times_ns) >= 1, (
                f"instance {instance.instance_id} has {len(status_times_ns)} "
                f"{IMInstance.InstanceStatus.Name(cur_status)} status"
            )
            status_time_ns = sorted(status_times_ns)[-1]
            if time.time_ns() - status_time_ns > (status_timeout_s * 1e9):
                updates[instance.instance_id] = IMInstanceUpdateEvent(
                    instance_id=instance.instance_id,
                    new_instance_status=new_status,
                    details=f"Timeout={status_timeout_s}s",
                )
                logger.info(
                    "Updating {}({}) with {}".format(
                        instance.instance_id,
                        IMInstance.InstanceStatus.Name(instance.status),
                        MessageToDict(updates[instance.instance_id]),
                    )
                )

        return updates

    @staticmethod
    def _warn_stuck_instances(
        instances: List[IMInstance],
        status: IMInstance.InstanceStatus,
        warn_interval_s: int,
        logger: logging.Logger,
    ):
        """Warn if any instance is stuck in a transient/unbounded status for too
        long.
        """
        for instance in instances:
            status_times_ns = InstanceUtil.get_status_transition_times_ns(
                instance, select_instance_status=status
            )
            assert len(status_times_ns) >= 1
            status_time_ns = sorted(status_times_ns)[-1]

            if time.time_ns() - status_time_ns > warn_interval_s * 1e9:
                logger.warning(
                    "Instance {}({}) is stuck in {} for {} seconds.".format(
                        instance.instance_id,
                        IMInstance.InstanceStatus.Name(instance.status),
                        IMInstance.InstanceStatus.Name(status),
                        (time.time_ns() - status_time_ns) // 1e9,
                    )
                )

    @staticmethod
    def _terminate_instances(instance_manager: InstanceManager):
        """
        Terminate instances with the below statuses:
            - RAY_STOPPED: ray was stopped on the cloud instance.
            - RAY_INSTALL_FAILED: ray installation failed on the cloud instance,
                we will not retry.
            - TERMINATION_FAILED: cloud provider failed to terminate the instance
                or timeout for termination happened, we will retry again.

        Args:
            instance_manager: The instance manager to reconcile.
        """

        im_instances, version = Reconciler._get_im_instances(instance_manager)
        updates = {}
        for instance in im_instances:
            if instance.status not in [
                IMInstance.RAY_STOPPED,
                IMInstance.RAY_INSTALL_FAILED,
                IMInstance.TERMINATION_FAILED,
            ]:
                continue

            # Terminate the instance.
            logger.info(
                f"Terminating instance {instance.instance_id} with status "
                f"{IMInstance.InstanceStatus.Name(instance.status)}"
            )
            updates[instance.instance_id] = IMInstanceUpdateEvent(
                instance_id=instance.instance_id,
                new_instance_status=IMInstance.TERMINATING,
            )

        Reconciler._update_instance_manager(instance_manager, version, updates)

    @staticmethod
    def _install_ray(
        instance_manager: InstanceManager,
        non_terminated_cloud_instances: Dict[CloudInstanceId, CloudInstance],
    )-> None:
        """
        Install ray on the allocated instances when it's ready (cloud instance
        should be running)

        This is needed if ray installation needs to be performed by
        the instance manager.

        Args:
            instance_manager: The instance manager to reconcile.
        """
        im_instances, version = Reconciler._get_im_instances(instance_manager)
        updates = {}
        for instance in im_instances:
            if instance.status != IMInstance.ALLOCATED:
                continue

            cloud_instance = non_terminated_cloud_instances.get(
                instance.cloud_instance_id
            )

            assert cloud_instance, (
                f"Cloud instance {instance.cloud_instance_id} is not found "
                "in non_terminated_cloud_instances."
            )

            if not cloud_instance.is_running:
                # It might still be pending (e.g. setting up ssh)
                continue

            # Install ray on the running cloud instance
            updates[instance.instance_id] = IMInstanceUpdateEvent(
                instance_id=instance.instance_id,
                new_instance_status=IMInstance.RAY_INSTALLING,
            )
            logger.info(
                "Updating {}({}) with {}".format(
                    instance.instance_id,
                    IMInstance.InstanceStatus.Name(instance.status),
                    MessageToDict(updates[instance.instance_id]),
                )
            )

        Reconciler._update_instance_manager(instance_manager, version, updates)


    def _handle_extra_cloud_instances(
        instance_manager: InstanceManager,
        non_terminated_cloud_instances: Dict[CloudInstanceId, CloudInstance],
    ):
        """
        Shut down extra cloud instances that are not managed by the instance manager.

        Since we have sync the IM states with the cloud provider's states in
        earlier step (`sync_from`), each non terminated cloud instance should either
        be:
            1. assigned to a newly ALLOCATED im instance
            2. already associated with an im instance that's running/terminating.

        Any cloud instance that's not managed by the IM should be considered leak.

        Args:
            instance_manager: The instance manager to reconcile.
            non_terminated_cloud_instances: The non-terminated cloud instances from
                the cloud provider.
        """
        instances, version = Reconciler._get_im_instances(instance_manager)

        cloud_instance_ids_managed_by_im = {
            instance.cloud_instance_id
            for instance in instances
            if instance.cloud_instance_id
        }

        leaked_cloud_instance_ids = []
        for cloud_instance_id, _ in non_terminated_cloud_instances.items():
            if cloud_instance_id in cloud_instance_ids_managed_by_im:
                continue

            leaked_cloud_instance_ids.append(cloud_instance_id)

        if not leaked_cloud_instance_ids:
            return

        # Update the IM with TERMINATING status for the leaked cloud instances.
        updates = {}

        for cloud_instance_id in leaked_cloud_instance_ids:
            updates[cloud_instance_id] = IMInstanceUpdateEvent(
                instance_id=InstanceUtil.random_instance_id(),  # Assign a new id.
                cloud_instance_id=cloud_instance_id,
                new_instance_status=IMInstance.TERMINATING,
                details="Leaked cloud instance",
            )

        Reconciler._update_instance_manager(instance_manager, version, updates)

        logger.warning(
            f"Terminating leaked cloud instances: {leaked_cloud_instance_ids}: no"
            " matching instance found in instance manager."
        )
